r"""
Build the per-frame contact-learning table from parsed resection data.

Usage:
    ros2 run visual_proprioception build_contact_dataset \
        --data-root ~/Resection_Data_Visual_Prop --out ~/aliss_runs/contact/frames.csv

Expected layout (output of the bag parser), at any depth under --data-root:
    [<session>/[rosbags/]]lesion_<L>/cut_<C>_rect_parsed/{resect_start,resect}/<seg>/
        endoscope/frameNNNNNN.png   rectified frames (a subset of the camera frames)
        kinematics.csv              one row per camera frame, 'filename' column

The lesion id is '<session>/lesion_<L>' (session = recording folder, e.g. one
bladder). A recording found twice (same kinematics timestamps) is used once:
the copy with the most approaches that have kinematics.

Approach segments (resect_start) without kinematics.csv or images are skipped. Each
approach is joined to the cutting segment (resect) that starts within
--join-gap seconds of its end. t_on is the laser-on edge (0 -> 1 in
/ves/right/measured_tool/data).

--labels depth (default): the target is the "cutting depth", where firing cuts
(the fiber may be ~1 mm above the tissue). The tip pose at a firing that cut
marks it; depth_to_cut_mm is how far short of it the fiber is along the beam.
A firing that the surgeon redid deeper within --retry-window s is a retry: its
frames are measured to the firing that followed.

    label  1  pre-fire     before laser-on, depth <= --depth-pos, lateral <= --lateral
    label  0  approaching  before laser-on, depth >= --depth-neg
    label -1  cutting      laser on, first --pos-cut s (reported, not trained on)
    label -1  excluded     everything else (kept for latency evaluation)

--labels laser: by time from laser-on (the surgeon's firing = contact):

    label  0  approaching  t - t_on < -neg_before, laser off
    label  1  pre-fire     -pos_pre <= t - t_on < 0
    label  1  cutting      0 <= t - t_on < pos_cut, laser on
    label -1  excluded     everything else

Both are written (laser_label/laser_kind hold the laser scheme).

Writes <out> (one row per frame) and <out>.json (the settings used).
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from .laser_spot_features import (FEATURE_NAMES, legacy_spot_area, measure_spot, project_tip,
                                  SPOT_PARAMS, SpotHistory, TipCheck, WINDOW_PARAMS)

LASER_COL = '/ves/right/measured_tool/data'
INSERTION_COL = '/ves/right/joint/measured_jp/position.2'
TIP_COLS = [f'/ves_smoother/right/tip_pose/pose.pose.position.{a}' for a in 'xyz']
QUAT_COLS = [f'/ves_smoother/right/tip_pose/pose.pose.orientation.{a}' for a in 'xyzw']
P_COLS = [f'/ves_camera/camera_info/p.{i}' for i in range(12)]
R_COLS = [f'/ves_camera/camera_info/r.{i}' for i in range(9)]
SIZE_COLS = ('/ves_camera/camera_info/width', '/ves_camera/camera_info/height')
CROP_CENTER = 'spot near the projected tip, else projected tip, else last spot (1.5 s)'


def _float(value):
    return float(value) if value not in ('', None) else float('nan')


def _load_segment(seg_dir):
    """Return kinematics rows that have an image, sorted by time (None if either is missing)."""
    kin_path = os.path.join(seg_dir, 'kinematics.csv')
    image_dir = os.path.join(seg_dir, 'endoscope')
    if not (os.path.exists(kin_path) and os.path.isdir(image_dir)):
        return None
    images = set(os.listdir(image_dir))
    with open(kin_path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['filename'] in images]
    rows.sort(key=lambda r: float(r['timestamp']))
    return rows


def _segment_ids(path):
    return sorted((int(s) for s in os.listdir(path) if s.isdigit())) if os.path.isdir(path) else []


def find_cuts(root):
    """
    Return [{lesion, session, cut, dir}] for every lesion_*/cut_*_rect_parsed under root.

    Accepts <root>/lesion_N/... and <root>/<session>/[rosbags/]lesion_N/...; the
    lesion id is the path from root without 'rosbags', e.g. '<session>/lesion_1'.
    """
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        name = os.path.basename(dirpath)
        lesion_dir = os.path.dirname(dirpath)
        if not (name.startswith('cut_') and name.endswith('_rect_parsed')
                and os.path.basename(lesion_dir).startswith('lesion_')):
            continue
        dirnames[:] = []
        parts = [p for p in os.path.relpath(lesion_dir, root).split(os.sep) if p != 'rosbags']
        found.append({'lesion': '/'.join(parts), 'session': parts[0] if len(parts) > 1 else '',
                      'cut': name.split('_rect_parsed')[0], 'dir': dirpath})
    return found


def _approach_stamps(cut_dir):
    """First kinematics timestamp of each approach segment: identifies the recording."""
    stamps = set()
    for seg in _segment_ids(os.path.join(cut_dir, 'resect_start')):
        kin_path = os.path.join(cut_dir, 'resect_start', str(seg), 'kinematics.csv')
        if os.path.exists(kin_path):
            with open(kin_path, newline='') as f:
                first = next(csv.DictReader(f), None)
            if first:
                stamps.add(round(float(first['timestamp']), 3))
    return stamps


def drop_duplicate_cuts(cuts):
    """
    Keep one copy of each recording found more than once (e.g. two parser exports).

    The copy with the most approach segments that have kinematics is kept; it takes
    the most specific lesion name, so both copies stay in the same CV group.
    """
    for c in cuts:
        c['stamps'] = _approach_stamps(c['dir'])
    kept, notes = [], []
    remaining = list(cuts)
    while remaining:
        group = [remaining.pop(0)]
        for other in list(remaining):
            if any(other['stamps'] & g['stamps'] for g in group):
                group.append(other)
                remaining.remove(other)
        best = max(group, key=lambda c: len(c['stamps']))
        if len(group) > 1:
            named = max(group, key=lambda c: (c['lesion'].count('/'), len(c['lesion'])))
            notes.append(
                f'same recording in {", ".join(c["dir"] for c in group)}; using '
                f'{best["dir"]} ({len(best["stamps"])} approaches with kinematics) '
                f'as {named["lesion"]}/{named["cut"]}')
            best = {**best, 'lesion': named['lesion'], 'session': named['session'],
                    'cut': named['cut']}
        kept.append(best)
    return kept, notes


def find_sequences(cut_dir, join_gap):
    """Yield (approach, cut_seg or None, frames) with frames = [(seg_dir, part, row), ...]."""
    cuts = {}
    for seg in _segment_ids(os.path.join(cut_dir, 'resect')):
        seg_dir = os.path.join(cut_dir, 'resect', str(seg))
        rows = _load_segment(seg_dir)
        if rows:
            cuts[seg] = (seg_dir, rows)
    for seg in _segment_ids(os.path.join(cut_dir, 'resect_start')):
        seg_dir = os.path.join(cut_dir, 'resect_start', str(seg))
        rows = _load_segment(seg_dir)
        if not rows:
            continue
        frames = [(seg_dir, 'approach', r) for r in rows]
        t_end = float(rows[-1]['timestamp'])
        follow = sorted(c for c, (_, cr) in cuts.items()
                        if 0.0 <= float(cr[0]['timestamp']) - t_end < join_gap)
        cut_seg = follow[0] if follow else None
        if cut_seg is not None:
            frames += [(cuts[cut_seg][0], 'cut', r) for r in cuts[cut_seg][1]]
        yield seg, cut_seg, frames


def laser_on_index(frames):
    """Index of the laser-on edge: the last one in the approach, else the first in the cut."""
    laser = np.array([_float(r[LASER_COL]) for _, _, r in frames])
    n_approach = sum(part == 'approach' for _, part, _ in frames)
    edges = np.flatnonzero((laser[1:] == 1) & (laser[:-1] == 0)) + 1
    in_approach = edges[edges < n_approach]
    if len(in_approach):
        return int(in_approach[-1])
    return int(edges[0]) if len(edges) else None


def row_tip(row):
    """Projected right fiber tip (pixels in the 540x540 frame) from a kinematics row, or None."""
    try:
        return project_tip([_float(row.get(c)) for c in TIP_COLS],
                           [_float(row[c]) for c in P_COLS], [_float(row[c]) for c in R_COLS],
                           *(_float(row[c]) for c in SIZE_COLS))
    except (KeyError, TypeError, ValueError):
        return None


def tip_pose(row):
    """(tip position in mm, unit beam direction = tip z axis), camera frame; None if missing."""
    try:
        p = 1000.0 * np.array([float(row[c]) for c in TIP_COLS])
        q = np.array([float(row[c]) for c in QUAT_COLS])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(p).all() and np.isfinite(q).all()) or np.linalg.norm(q) < 1e-9:
        return None
    x, y, z, w = q / np.linalg.norm(q)
    return p, np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def firing_targets(jobs, labels):
    """
    Return ({job index: (position, beam) of the firing that cut}, {retried job indices}).

    A firing counts as retried when the next firing in the same cut starts within
    retry_s of it, with the tip more than retry_deeper_mm deeper along the beam and less
    than lateral_mm to the side: the surgeon judged the first depth too shallow. A
    retried firing takes the target of the firing that followed it.
    """
    info = {}
    for k, job in enumerate(jobs):
        frames = job[5]
        edge = laser_on_index(frames)
        pose = tip_pose(frames[edge][2]) if edge is not None else None
        if pose is not None:
            info[k] = (float(frames[edge][2]['timestamp']), *pose)
    by_cut = defaultdict(list)
    for k in info:
        by_cut[(jobs[k][1], jobs[k][2])].append(k)
    targets, retried = {}, set()
    for ks in by_cut.values():
        ks.sort(key=lambda k: info[k][0])
        for a, b in zip(ks[:-1], ks[1:]):
            (ta, pa, beam), (tb, pb, _) = info[a], info[b]
            along = float((pb - pa) @ beam)
            side = float(np.linalg.norm(pb - pa - along * beam))
            if (tb - ta <= labels['retry_s'] and along > labels['retry_deeper_mm']
                    and side < labels['lateral_mm']):
                retried.add(a)
        for i in range(len(ks) - 1, -1, -1):
            k = ks[i]
            retry = k in retried and i + 1 < len(ks)
            targets[k] = targets[ks[i + 1]] if retry else info[k][1:]
    return targets, retried


def label_frame(rel, laser, labels, depth=math.nan, lateral=math.nan):
    """
    Return (label, kind) for one frame.

    laser scheme: by time from laser-on (see the module docstring).
    depth scheme: before firing, 'pre-fire' (1) = within depth_pos_mm of the cutting
    depth along the beam and lateral_mm sideways; 'approaching' (0) = depth_neg_mm or
    more short of it. Firing frames are never trained on ('cutting' = first pos_cut s).
    """
    if labels.get('scheme', 'laser') == 'laser':
        if rel < -labels['neg_before'] and laser == 0:
            return 0, 'approaching'
        if -labels['pos_pre'] <= rel < 0:
            return 1, 'pre-fire'
        if 0 <= rel < labels['pos_cut'] and laser == 1:
            return 1, 'cutting'
        return -1, 'excluded'
    if rel >= 0 or laser == 1:
        return -1, 'cutting' if 0 <= rel < labels['pos_cut'] and laser == 1 else 'excluded'
    if depth <= labels['depth_pos_mm'] and lateral <= labels['lateral_mm']:
        return 1, 'pre-fire'
    if depth >= labels['depth_neg_mm']:
        return 0, 'approaching'
    return -1, 'excluded'


def process_sequence(job):
    """Compute the table rows for one approach (+ following cut); runs in a worker."""
    (session, lesion, cut, approach, cut_seg, frames, labels, spot_params, window,
     target, retried) = job
    edge = laser_on_index(frames)
    if edge is None:
        return [], 'no laser-on edge, skipped'
    depth_scheme = labels.get('scheme', 'laser') == 'depth'
    if depth_scheme and target is None:
        return [], 'no tip pose at laser-on, skipped'
    t_on = float(frames[edge][2]['timestamp'])
    history = SpotHistory(**window)
    tip_check = TipCheck()
    out = []
    for seg_dir, part, r in frames:
        t = float(r['timestamp'])
        laser = _float(r[LASER_COL])
        depth = lateral = math.nan
        pose = tip_pose(r) if target is not None else None
        if pose is not None:
            offset = target[0] - pose[0]
            depth = float(offset @ target[1])
            lateral = float(np.linalg.norm(offset - depth * target[1]))
        label, kind = label_frame(t - t_on, laser, labels, depth, lateral)
        laser_label, laser_kind = label_frame(t - t_on, laser, {**labels, 'scheme': 'laser'})
        path = os.path.abspath(os.path.join(seg_dir, 'endoscope', r['filename']))
        image = cv2.imread(path)
        if image is None:
            continue
        tip = row_tip(r)
        try:
            if tip is not None and not tip_check.update(t, [_float(r[c]) for c in TIP_COLS]):
                tip = None
        except KeyError:
            tip = None
        spot, _ = measure_spot(image, spot_params, tip)
        feats, center = history.update(t, spot, tip)
        source = ('spot' if spot['detected'] else 'tip' if tip is not None
                  else 'last spot' if center is not None else 'image centre')
        row = {
            'session': session, 'lesion': lesion, 'cut': cut, 'approach': approach,
            'sequence': f'{lesion}/{cut}/{approach}',
            'cut_segment': '' if cut_seg is None else cut_seg, 'part': part, 'image': path,
            't': t, 't_rel': t - t_on, 'laser': laser, 'label': label, 'kind': kind,
            'laser_label': laser_label, 'laser_kind': laser_kind,
            'depth_to_cut_mm': depth, 'lateral_mm': lateral, 'retried': int(retried),
            'insertion_mm': 1000.0 * _float(r.get(INSERTION_COL)),
            'legacy_area': legacy_spot_area(image),
            'spot_u': spot['u'], 'spot_v': spot['v'], 'spot_area': spot['area'],
            'spot_major': spot['major'], 'spot_minor': spot['minor'],
            'crop_u': center[0] if center else float('nan'),
            'crop_v': center[1] if center else float('nan'),
            'tip_u': tip[0] if tip else float('nan'),
            'tip_v': tip[1] if tip else float('nan'),
            'crop_source': source,
        }
        for axis, col in zip('xyz', TIP_COLS):
            row[f'tip_{axis}_mm'] = 1000.0 * _float(r.get(col))
        row.update(feats)
        out.append(row)
    kinds = [row['kind'] for row in out]
    where = 'approach' if frames[edge][1] == 'approach' else 'cut'
    counts = ', '.join(f'{k}={kinds.count(k)}'
                       for k in ('approaching', 'pre-fire', 'cutting', 'excluded'))
    return out, f'laser-on in {where}{"; retried" if retried else ""}; {counts}'


def _worker_init():
    cv2.setNumThreads(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True,
                        help='folder searched for lesion_*/cut_*_rect_parsed directories')
    parser.add_argument('--out', required=True, help='output CSV path')
    parser.add_argument('--join-gap', type=float, default=0.3,
                        help='max s between approach end and cut start to join them')
    parser.add_argument('--neg-before', type=float, default=0.5,
                        help='approach frames earlier than this before laser-on are label 0')
    parser.add_argument('--pos-pre', type=float, default=0.15,
                        help='frames this close before laser-on are label 1')
    parser.add_argument('--pos-cut', type=float, default=1.0,
                        help='cutting frames this soon after laser-on are label 1')
    parser.add_argument('--labels', choices=('depth', 'laser'), default='depth',
                        help='depth: label by distance to the cutting depth (kinematics); '
                             'laser: by time from laser-on')
    parser.add_argument('--depth-pos', type=float, default=1.0,
                        help='depth labels: within this many mm of the cutting depth = 1')
    parser.add_argument('--depth-neg', type=float, default=2.0,
                        help='depth labels: this many mm or more short of it = 0')
    parser.add_argument('--lateral', type=float, default=2.0,
                        help='depth labels: max mm sideways of the firing position for 1, '
                             'and for a refiring to count as a retry')
    parser.add_argument('--retry-window', type=float, default=8.0,
                        help='a refiring within this many s of laser-on can be a retry')
    parser.add_argument('--retry-deeper', type=float, default=0.5,
                        help='a retry fires at least this many mm deeper along the beam')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args(argv)

    root = os.path.expanduser(args.data_root)
    labels = {'scheme': args.labels, 'neg_before': args.neg_before, 'pos_pre': args.pos_pre,
              'pos_cut': args.pos_cut, 'depth_pos_mm': args.depth_pos,
              'depth_neg_mm': args.depth_neg, 'lateral_mm': args.lateral,
              'retry_s': args.retry_window, 'retry_deeper_mm': args.retry_deeper}
    cuts, notes = drop_duplicate_cuts(find_cuts(root))
    for note in notes:
        print(f'NOTE: {note}')
    jobs = []
    for c in cuts:
        for approach, cut_seg, frames in find_sequences(c['dir'], args.join_gap):
            jobs.append((c['session'], c['lesion'], c['cut'], approach, cut_seg, frames, labels,
                         SPOT_PARAMS, WINDOW_PARAMS))
    targets, retried = firing_targets(jobs, labels)
    jobs = [job + (targets.get(k), k in retried) for k, job in enumerate(jobs)]
    if not jobs:
        sys.exit(f'No approach segments with kinematics.csv found under {root}')

    started = time.time()
    table = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for job, (rows, report) in zip(jobs, pool.map(process_sequence, jobs)):
            lesion, cut, approach, cut_seg = job[1:5]
            joined = f' + cut {cut_seg}' if cut_seg is not None else ''
            print(f'{lesion}/{cut} approach {approach}{joined}: {report}')
            table.extend(rows)
    if not table:
        sys.exit('No approach had a laser-on edge; nothing written.')

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    columns = list(table[0].keys())
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)

    kinds = [r['kind'] for r in table]
    summary = {k: kinds.count(k) for k in ('approaching', 'pre-fire', 'cutting', 'excluded')}
    sequences = sorted({r['sequence'] for r in table})
    meta = {
        'data_root': os.path.abspath(root), 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'labels': labels, 'join_gap': args.join_gap, 'spot_params': SPOT_PARAMS,
        'window': WINDOW_PARAMS, 'feature_names': list(FEATURE_NAMES), 'crop_center': CROP_CENTER,
        'frames': summary, 'sequences': len(sequences), 'retried_firings': len(retried),
        'lesions': sorted({r['lesion'] for r in table}),
        'sessions': sorted({r['session'] for r in table}),
        'duplicates': notes,
    }
    with open(out + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    detected = [r for r in table if r['kind'] != 'excluded']
    seen = sum(r['detected'] for r in detected) / max(len(detected), 1)
    print(f'\n{len(table)} frames from {len(sequences)} approaches in '
          f'{len(meta["lesions"])} lesion(s), {len(meta["sessions"])} session(s) -> {out}  '
          f'({time.time() - started:.0f} s)')
    print('  ' + ', '.join(f'{k}={v}' for k, v in summary.items()))
    print(f'  spot detected in {seen:.0%} of labelled frames')
    sources = [r['crop_source'] for r in detected]
    print('  crop centre: ' + ', '.join(f'{k} {sources.count(k) / max(len(sources), 1):.0%}'
                                        for k in ('spot', 'tip', 'last spot', 'image centre')))
    if math.isclose(seen, 0.0):
        print('  WARNING: no spots detected; check SPOT_PARAMS against this camera.')


if __name__ == '__main__':
    main()
