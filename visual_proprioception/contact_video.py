r"""
Render a video of held-out contact predictions from a train_contact_model run.

Usage:
    ros2 run visual_proprioception render_contact_video \
        --run-dir ~/aliss_runs/contact/run1 --model combined

Plays the approaches back to back at real-time speed. Every frame shows its
out-of-fold prediction: the score from the cross-validation model that was
trained without that approach, so the video shows performance on unseen data.

    top right  model decision: CONTACT (P >= confirm), SUSPECTED (P >= suspect)
               or NO CONTACT, with P(contact)
    bottom     ground truth from the laser-on signal and the probability trace

P is smoothed over --smooth s (default: the model's smooth_sec, as the node does).
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contact_model import CausalMean

FONTS = {
    False: ['/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
    True: ['/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'],
}
PANEL_H = 190
BG = (22, 24, 29)
TEXT = (235, 237, 240)
MUTED = (150, 156, 168)
STATE_COLORS = {'CONTACT': ((31, 157, 85), TEXT), 'SUSPECTED': ((230, 162, 0), (20, 20, 20)),
                'NO CONTACT': ((58, 63, 75), TEXT)}
GT_NONE, GT_CONTACT, GT_UNLABELLED = (66, 133, 244), (245, 130, 32), (95, 100, 110)


def font(size, bold=False):
    for path in FONTS[bold]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ground_truth(kind, t_rel, laser, labels=None):
    labels = labels or {}
    depth = labels.get('scheme') == 'depth'
    if kind == 'approaching':
        short = f'too shallow (>= {labels.get("depth_neg_mm", 2):g} mm short)'
        return (short if depth else 'no contact (approaching, laser off)'), GT_NONE
    if kind == 'pre-fire':
        return ((f'at cutting depth (<= {labels.get("depth_pos_mm", 1):g} mm)', GT_CONTACT)
                if depth else ('contact (laser fires within 0.15 s)', GT_CONTACT))
    if depth and laser == 1:
        return 'laser firing (not scored)', GT_UNLABELLED
    if t_rel < 0:
        return (('in between (not scored)', GT_UNLABELLED) if depth
                else ('not labelled (0.15-0.5 s before firing)', GT_UNLABELLED))
    if laser == 1:
        return 'contact (laser on, cutting)', GT_CONTACT
    return 'not labelled (laser off)', GT_UNLABELLED


def load_run(run_dir, model):
    """Return (sequences {name: [frame dicts]}, model meta, metrics) for one trained model."""
    model_dir = os.path.join(run_dir, model)
    with open(os.path.join(model_dir, 'model.json')) as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, 'oof_predictions.csv'), newline='') as f:
        preds = list(csv.DictReader(f))
    with open(meta['trained_on']['frames'], newline='') as f:
        frames = list(csv.DictReader(f))
    if len(preds) != len(frames):
        sys.exit('oof_predictions.csv and the frame table differ in length; '
                 'was the table rebuilt after training?')
    sequences = {}
    for p, r in zip(preds, frames):
        if p['sequence'] != r['sequence'] or abs(float(p['t']) - float(r['t'])) > 1e-3:
            sys.exit('oof_predictions.csv does not match the frame table row by row.')
        sequences.setdefault(r['sequence'], []).append({
            'image': r['image'], 't': float(r['t']), 't_rel': float(r['t_rel']),
            'kind': r['kind'], 'laser': float(r['laser']), 'prob': float(p['prob']),
            'depth': float(r.get('depth_to_cut_mm') or 'nan')})
    for rows in sequences.values():
        rows.sort(key=lambda row: row['t'])
    metrics = {}
    path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
    return sequences, meta, metrics


def pretty(name):
    """Shorten 'bladder_tele_vu_00_3fa7adea/lesion_1/cut_1' to 'vu_00 / lesion_1 / cut_1'."""
    parts = [re.sub(r'^bladder_tele_|_[0-9A-Za-z]{8}$', '', p) for p in name.split('/')]
    return ' / '.join(parts)


def smooth(sequences, window):
    for rows in sequences.values():
        mean = CausalMean(window)
        for row in rows:
            row['prob'] = mean.update(row['t'], row['prob'])


def sequence_key(name):
    parts = name.split('/')
    return parts[:-1], int(parts[-1]) if parts[-1].isdigit() else parts[-1]


def centered(draw, y, text, fnt, fill, width):
    while draw.textlength(text, font=fnt) > width - 40 and fnt.size > 10:
        fnt = fnt.font_variant(size=fnt.size - 1)
    w = draw.textlength(text, font=fnt)
    draw.text(((width - w) / 2, y), text, font=fnt, fill=fill)


class Renderer:
    def __init__(self, width, thresholds, labels=None):
        self.labels = labels
        self.depth = (labels or {}).get('scheme') == 'depth'
        depth_names = {'CONTACT': 'AT CUT DEPTH', 'SUSPECTED': 'CLOSE',
                       'NO CONTACT': 'TOO SHALLOW'}
        self.names = depth_names if self.depth else {k: k for k in STATE_COLORS}
        self.p_name = 'P(at cut depth)' if self.depth else 'P(contact)'
        self.w = width
        self.thresholds = thresholds
        self.f = {k: font(s, b) for k, (s, b) in {
            'state': (30, True), 'prob': (17, False), 'label': (16, True), 'gt': (18, False),
            'small': (13, False), 'title': (34, True), 'body': (19, False)}.items()}

    def state(self, prob):
        if prob >= self.thresholds['confirm']:
            return 'CONTACT'
        return 'SUSPECTED' if prob >= self.thresholds['suspect'] else 'NO CONTACT'

    def image(self, path, dim=1.0):
        bgr = cv2.imread(path)
        h = int(round(bgr.shape[0] * self.w / bgr.shape[1])) // 2 * 2
        bgr = cv2.resize(bgr, (self.w, h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if dim != 1.0:
            rgb = (rgb * dim).astype(np.uint8)
        return Image.fromarray(rgb)

    def frame(self, rows, i, title):
        row = rows[i]
        top = self.image(row['image'])
        ih = top.height
        canvas = Image.new('RGB', (self.w, ih + PANEL_H), BG)
        canvas.paste(top, (0, 0))
        d = ImageDraw.Draw(canvas, 'RGBA')

        # top left: which approach
        tw = d.textlength(title, font=self.f['label'])
        d.rounded_rectangle((12, 12, 32 + tw, 44), 8, fill=(0, 0, 0, 150))
        d.text((22, 17), title, font=self.f['label'], fill=TEXT)

        # top right: model decision
        state = self.state(row['prob'])
        bg, fg = STATE_COLORS[state]
        bw, bh = 250, 80
        x0, y0 = self.w - bw - 14, 14
        d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 12, fill=bg + (235,))
        shown, sf = self.names[state], self.f['state']
        while d.textlength(shown, font=sf) > bw - 16 and sf.size > 12:
            sf = sf.font_variant(size=sf.size - 1)
        sw = d.textlength(shown, font=sf)
        d.text((x0 + (bw - sw) / 2, y0 + 7), shown, font=sf, fill=fg)
        ptxt = f'{self.p_name} = {row["prob"]:.2f}'
        pw = d.textlength(ptxt, font=self.f['prob'])
        d.text((x0 + (bw - pw) / 2, y0 + 50), ptxt, font=self.f['prob'], fill=fg)

        # bottom panel: ground truth line
        py = ih
        gt_text, gt_color = ground_truth(row['kind'], row['t_rel'], row['laser'], self.labels)
        if self.depth and math.isfinite(row['depth']) and row['t_rel'] < 0:
            mm = row['depth']
            where = f'fiber {mm:.1f} mm short' if mm >= 0 else f'fiber {-mm:.1f} mm past'
            gt_text = f"{gt_text.split(' (')[0]}  ·  {where}"
        gt_text = f'Ground truth: {gt_text}'
        d.ellipse((18, py + 19, 30, py + 31), fill=gt_color)
        d.text((40, py + 12), gt_text, font=self.f['gt'], fill=TEXT)
        ttxt = f't = {row["t_rel"]:+.2f} s from laser-on'
        if 40 + d.textlength(gt_text, font=self.f['gt']) + 24 + d.textlength(
                ttxt, font=self.f['prob']) > self.w - 16:
            ttxt = f't = {row["t_rel"]:+.2f} s'
        d.text((self.w - 16 - d.textlength(ttxt, font=self.f['prob']), py + 14), ttxt,
               font=self.f['prob'], fill=MUTED)

        # probability trace
        left, right, ptop, pbot = 52, self.w - 18, py + 58, py + 146
        t0, t1 = rows[0]['t_rel'], rows[-1]['t_rel']
        span = max(t1 - t0, 1e-6)

        def x(t):
            return left + (t - t0) / span * (right - left)

        def y(p):
            return pbot - p * (pbot - ptop)

        d.rectangle((left, ptop, right, pbot), outline=(60, 64, 72))
        d.text((left, py + 40), self.p_name, font=self.f['small'], fill=MUTED)
        for name, color in (('confirm', STATE_COLORS['CONTACT'][0]),
                            ('suspect', STATE_COLORS['SUSPECTED'][0])):
            thr = self.thresholds[name]
            for xs in range(left, right, 10):
                d.line((xs, y(thr), min(xs + 5, right), y(thr)), fill=color, width=1)
            d.text((16, y(thr) - 9), f'{thr:.1f}', font=self.f['small'], fill=color)
        for j, r in enumerate(rows):
            xe = x(rows[j + 1]['t_rel']) if j + 1 < len(rows) else right
            d.rectangle((x(r['t_rel']), pbot + 6, xe, pbot + 14),
                        fill=ground_truth(r['kind'], r['t_rel'], r['laser'], self.labels)[1])
        if t0 <= 0 <= t1:
            d.line((x(0), ptop, x(0), pbot + 14), fill=TEXT, width=1)
            d.text((x(0) + 4, py + 40), 'laser on', font=self.f['small'], fill=TEXT)
        pts = [(x(r['t_rel']), y(r['prob'])) for r in rows[:i + 1]]
        if len(pts) > 1:
            d.line(pts, fill=TEXT, width=2, joint='curve')
        cx, cy = pts[-1]
        d.line((cx, ptop, cx, pbot), fill=MUTED + (160,), width=1)
        d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=bg, outline=TEXT)
        lx = left
        legend = ((('too shallow', GT_NONE), ('at cutting depth', GT_CONTACT),
                   ('not scored', GT_UNLABELLED)) if self.depth else
                  (('no contact', GT_NONE), ('contact', GT_CONTACT),
                   ('not labelled', GT_UNLABELLED)))
        for label, color in legend:
            d.rectangle((lx, py + 170, lx + 10, py + 180), fill=color)
            d.text((lx + 15, py + 166), label, font=self.f['small'], fill=MUTED)
            lx += 30 + d.textlength(label, font=self.f['small'])
        return canvas

    def card(self, lines, size, background=None):
        canvas = background.copy() if background else Image.new('RGB', size, BG)
        d = ImageDraw.Draw(canvas)
        total = sum(h for _, _, h in lines)
        yy = (size[1] - total) / 2
        for text, fnt, h in lines:
            centered(d, yy, text, self.f[fnt], TEXT if fnt == 'title' else MUTED, size[0])
            yy += h
        return canvas


def summary_lines(metrics, model, n_approaches, depth=False):
    res = metrics.get('results', {})
    if model not in res:
        return []
    model_name = model.split(' ~')[0]
    m, base = res[model]['confirm'], res.get('current rule', {}).get('confirm')
    if depth:
        return [
            (f'Held-out results, {model_name} model, all {m["approaches"]} approaches',
             'title', 60),
            (f'Flagged at cutting depth before firing: {m["before_fire"]:.0%} of approaches',
             'body', 34),
            (f'Flagged by 1 s after laser-on: {m["by_1s"]:.0%}', 'body', 34),
            (f'False alarm while 2+ mm short: {m["false_alarm"]:.0%} of approaches', 'body', 50),
            ('Each bladder session was scored by a model trained without it.', 'body', 30),
        ]

    def cmp(key):
        return f'   (current area rule: {base[key]:.0%})' if base else ''
    units = {'session': 'bladder sessions', 'lesion': 'lesions'}
    unit = units.get(metrics.get('group_by'), 'approaches')
    measurable = m.get('before_fire_measurable', m['approaches'])
    return [
        (f'Held-out results, {model_name} model, {n_approaches} approaches', 'title', 60),
        (f'Detected before laser-on: {m["before_fire"]:.0%} of the {measurable} approaches '
         f'with a frame just before firing{cmp("before_fire")}', 'body', 34),
        (f'Detected within 1 s of laser-on: {m["by_1s"]:.0%}{cmp("by_1s")}', 'body', 34),
        (f'False alarm while approaching: {m["false_alarm"]:.0%}{cmp("false_alarm")}',
         'body', 50),
        (f'Cross-validation held out whole {unit}.', 'body', 30),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='--out-dir of train_contact_model')
    parser.add_argument('--model', default='combined',
                        choices=('spot', 'image', 'combined', 'temporal'))
    parser.add_argument('--out', help='output .mp4 (default <run-dir>/<model>_heldout.mp4)')
    parser.add_argument('--sequences', nargs='*', help='only these approaches, e.g. '
                        'lesion_1/cut_1/19 (default: all)')
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor')
    parser.add_argument('--smooth', type=float,
                        help="causal mean of P over this many s (default: the model's value)")
    args = parser.parse_args(argv)

    run_dir = os.path.expanduser(args.run_dir)
    sequences, meta, metrics = load_run(run_dir, args.model)
    if meta.get('target') == 'depth':
        sys.exit('render_contact_video shows P(contact); models trained with --target depth '
                 '(mm to the cutting depth) are not supported yet.')
    window = args.smooth if args.smooth is not None else float(meta.get('smooth_sec') or 0.0)
    smooth(sequences, window)
    shown = f'{args.model} ~{window:g}s' if window > 0 else args.model
    names = sorted(args.sequences or sequences, key=sequence_key)
    missing = [n for n in names if n not in sequences]
    if missing:
        sys.exit(f'Unknown sequences: {missing}')
    thresholds = {'suspect': 0.6, 'confirm': 0.8, **meta.get('thresholds', {})}
    render = Renderer(args.width // 2 * 2, thresholds, meta.get('labels'))

    first = render.frame(sequences[names[0]], 0, '')
    size = first.size
    out = os.path.expanduser(args.out or os.path.join(run_dir, f'{args.model}_heldout.mp4'))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), args.fps, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, size)
    clock = 0.0

    def emit(img, seconds):
        nonlocal clock
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        n = round((clock + seconds) * args.fps) - round(clock * args.fps)
        for _ in range(max(n, 1)):
            writer.write(bgr)
        clock += seconds

    units = {'session': 'bladder session', 'lesion': 'lesion'}
    unit = units.get(metrics.get('group_by'), 'approach')
    labels = meta.get('labels') or {}
    if render.depth:
        intro = [
            ('Laser cutting-depth detection on unseen data', 'title', 60),
            (f'{args.model} model{f", P averaged over {window:g} s" if window > 0 else ""}, '
             f'{len(names)} approaches, real-time playback', 'body', 34),
            (f'Each {unit} is scored by a cross-validation model trained without it.',
             'body', 34),
            (f'Top right: model output. AT CUT DEPTH if P >= {thresholds["confirm"]:.1f}, '
             f'CLOSE if P >= {thresholds["suspect"]:.1f}', 'body', 34),
            ('Bottom: ground truth from robot kinematics:', 'body', 30),
            (f'at cutting depth = within {labels.get("depth_pos_mm", 1):g} mm of where the '
             f'surgeon fired', 'body', 34),
        ]
        emit(render.card(intro, size), 5.0)
    else:
        emit(render.card([
            ('Laser-tissue contact detection on unseen data', 'title', 60),
            (f'{args.model} model{f", P averaged over {window:g} s" if window > 0 else ""}, '
             f'{len(names)} approaches, real-time playback', 'body', 34),
            (f'Each {unit} is scored by a cross-validation model trained without it.',
             'body', 34),
            (f'Top right: model output. CONTACT if P >= {thresholds["confirm"]:.1f}, '
             f'SUSPECTED if P >= {thresholds["suspect"]:.1f}', 'body', 34),
            ('Bottom: ground truth from the laser-on signal (surgeon fired in contact)',
             'body', 34),
        ], size), 4.0)

    for k, name in enumerate(names):
        rows = sequences[name]
        title = f'Approach {name.split("/")[-1]}  ({k + 1}/{len(names)})  held out'
        background = render.frame(rows, 0, title)
        background = Image.eval(background, lambda v: int(v * 0.3))
        emit(render.card([(f'Approach {name.split("/")[-1]}', 'title', 52),
                          (f'{k + 1} of {len(names)}  ·  {pretty(name.rsplit("/", 1)[0])}',
                           'body', 30)],
                         size, background), 1.2)
        for i, row in enumerate(rows):
            dt = rows[i + 1]['t'] - row['t'] if i + 1 < len(rows) else 0.6
            emit(render.frame(rows, i, title), min(max(dt, 1.0 / args.fps), 0.5) / args.speed)
        print(f'{name}: {len(rows)} frames')

    lines = (summary_lines(metrics, shown, len(names), depth=True) if render.depth
             else summary_lines(metrics, shown, len(names)) if not args.sequences else [])
    if lines:
        emit(render.card(lines, size), 6.0)
    writer.release()
    print(f'{out}  ({clock:.0f} s, {size[0]}x{size[1]})')


if __name__ == '__main__':
    main()
