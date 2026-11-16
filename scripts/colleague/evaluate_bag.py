"""
Compare our model and the colleague's on one bag, with depth labels from the tip pose.

Input: the CSV from score_bag.py. Labels follow build_contact_dataset: each laser 0->1
edge is a firing; its reference is the tip at laser-on (position, beam = tip z axis). A
firing followed within 8 s by one >0.5 mm deeper and <2 mm sideways is a retry and
takes the later reference. Frames between the previous laser-off and this laser-on are
the approach: depth = (p_ref - p) . beam; yes if depth <= 1 mm and lateral <= 2 mm, no if
depth >= 2 mm. Only frames with a usable tip pose (TipCheck) count, and both models are
compared on the frames where the colleague's model has a score.

Reports AUC at 0.5-2 mm cutoffs, laser-on vs far AUC, firings flagged in the 1 s before
laser-on at each model's own threshold and at matched false-alarm rates, and a bootstrap
interval over firings for the AUC difference at 1 mm.

Usage: evaluate_bag.py <score_bag.csv>
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from visual_proprioception.contact_train import auc  # noqa: E402


def beam(quat):
    """Return the tip frame's unit z axis (the laser beam direction) for a quaternion."""
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def causal_mean(t, p, window=0.3, gap=1.0):
    """Mean of the finite values over the last `window` s; restarts after a `gap` s break."""
    out, start, seg = np.empty_like(p), 0, 0
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(p))])
    cnt = np.concatenate([[0], np.cumsum(np.isfinite(p))])
    for i in range(len(p)):
        if i and t[i] - t[i - 1] > gap:
            seg = i
        while start < i and (t[start] < t[i] - window or start < seg):
            start += 1
        n = cnt[i + 1] - cnt[start]
        out[i] = (csum[i + 1] - csum[start]) / n if n else np.nan
    return out


def depth_labels(t, laser, pos_mm, quats, ok):
    """Per-frame depth / lateral offset to the firing that ends each approach."""
    onsets = np.flatnonzero(laser & ~np.r_[False, laser[:-1]])
    offs = np.flatnonzero(~laser & np.r_[False, laser[:-1]])
    fires = [e for e in onsets if ok[e]]
    ref = {e: (pos_mm[e], beam(quats[e])) for e in fires}
    retried = set()
    for a, b in zip(fires[:-1], fires[1:]):
        (pa, ba), (pb, _) = ref[a], ref[b]
        along = float((pb - pa) @ ba)
        side = float(np.linalg.norm(pb - pa - along * ba))
        if t[b] - t[a] <= 8.0 and along > 0.5 and side < 2.0:
            retried.add(a)
    target = {}
    for k in range(len(fires) - 1, -1, -1):
        e = fires[k]
        target[e] = target[fires[k + 1]] if e in retried and k + 1 < len(fires) else ref[e]
    depth = np.full(len(t), np.nan)
    lateral = np.full(len(t), np.nan)
    firing_of = np.full(len(t), -1)
    for e in fires:
        prev_off = offs[offs < e]
        lo = prev_off[-1] if len(prev_off) else 0
        p_ref, b_ref = target[e]
        idx = np.arange(lo, e)
        idx = idx[ok[idx] & ~laser[idx]]
        d = p_ref - pos_mm[idx]
        along = d @ b_ref
        depth[idx] = along
        lateral[idx] = np.linalg.norm(d - along[:, None] * b_ref, axis=1)
        firing_of[idx] = e
    return onsets, fires, retried, depth, lateral, firing_of


def flagged_before(p, t, onsets, both, thr):
    """Whether each firing has a score >= thr in the 1 s before laser-on."""
    out = []
    for e in onsets:
        window = (t >= t[e] - 1) & (t < t[e])
        if both[window].any():
            out.append(np.nanmax(p[window], initial=-1) >= thr)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('csv', help='output of score_bag.py')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()

    with open(args.csv, newline='') as f:
        rows = list(csv.DictReader(f))
    col = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    t, laser = col['t'], col['laser'] > 0.5
    pos_mm = np.stack([col['x'], col['y'], col['z']], 1) * 1000.0
    quats = np.stack([col['qx'], col['qy'], col['qz'], col['qw']], 1)
    ok = (col['tip_ok'] > 0) & np.isfinite(pos_mm).all(1)
    onsets, fires, retried, depth, lateral, firing_of = depth_labels(t, laser, pos_mm, quats, ok)
    yes = (depth <= 1.0) & (lateral <= 2.0)
    no = depth >= 2.0
    both = np.isfinite(col['theirs'])
    print(f'{len(t)} frames, {len(onsets)} firings ({len(fires)} with a usable tip pose, '
          f'{len(retried)} retried); approach frames labelled: {int(yes.sum())} yes, '
          f'{int(no.sum())} no')
    print(f'colleague model has a score on {both.mean():.0%} of frames '
          f'(needs the projected tip in the image); ours on 100%')
    print(f'compared on frames where both have a score: {int((yes & both).sum())} yes, '
          f'{int((no & both).sum())} no\n')

    theirs_smooth = causal_mean(t, col['theirs'])
    models = {
        'ours (raw)': col['ours_raw'],
        'ours (0.3 s smoothed, as live)': col['ours'],
        'colleague (raw)': col['theirs'],
        'colleague (0.3 s smoothed)': theirs_smooth,
    }
    print(f'{"model":36s}' + ''.join(f'{f"AUC {c}mm":>10s}' for c in (0.5, 1, 1.5, 2))
          + f'{"AUC cutting":>13s}')
    for name, p in models.items():
        cells = []
        for c in (0.5, 1.0, 1.5, 2.0):
            cells.append(auc(p, (depth <= c) & (lateral <= 2.0) & both,
                             (depth >= c + 1.0) & both))
        cells.append(auc(p, laser & both, no & both))
        print(f'{name:36s}' + ''.join(f'{v:10.2f}' for v in cells[:4]) + f'{cells[4]:13.2f}')

    print("\nAt each model's own threshold (ours 0.8 smoothed, colleague 0.5 raw), firings "
          'with a score in the 1 s before:')
    for name, p, thr in (('ours', col['ours'], 0.8), ('colleague', col['theirs'], 0.5)):
        det = flagged_before(p, t, onsets, both, thr)
        print(f'  {name:10s} flagged before {sum(det)} of {len(det)} firings '
              f'({np.mean(det):.0%}); "no" frames (>= 2 mm short) flagged '
              f'{np.mean(p[no & both] >= thr):.0%}; all laser-off frames flagged '
              f'{np.nanmean(p[~laser & both] >= thr):.0%}')

    print('\nMatched false alarms: threshold set so each flags the same share of "no" frames')
    for share in (0.05, 0.10, 0.20):
        out = []
        for name, p in (('ours', col['ours']), ('colleague', col['theirs'])):
            thr = np.quantile(p[no & both], 1 - share)
            det = flagged_before(p, t, onsets, both, thr)
            out.append(f'{name}: firings flagged {np.mean(det):.0%}, '
                       f'"yes" frames {np.mean(p[yes & both] >= thr):.0%}')
        print(f'  {share:.0%} of "no" frames flagged -> ' + ';  '.join(out))

    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(firing_of == e) for e in fires if (firing_of == e).any()]
    a, b = col['ours'], theirs_smooth
    diffs = []
    for _ in range(args.bootstrap):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        m_yes, m_no = (yes & both)[idx], (no & both)[idx]
        diffs.append(auc(a[idx], m_yes, m_no) - auc(b[idx], m_yes, m_no))
    print(f'\nAUC at 1 mm, ours - colleague (both smoothed): '
          f'{auc(a, yes & both, no & both) - auc(b, yes & both, no & both):+.3f}, '
          f'95% interval over firings [{np.percentile(diffs, 2.5):+.3f}, '
          f'{np.percentile(diffs, 97.5):+.3f}]')


if __name__ == '__main__':
    main()
