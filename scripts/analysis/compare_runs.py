"""
Compare the training runs on the same held-out frames with the depth labels.

Runs (in <work_dir>, as made by contact_train): runA_seed0 (laser-timing labels),
runD_seed* (yes/no depth labels at 1 mm; the shipped model is runD_seed0/combined) and
runR_seed0 (distance in mm). Every run's out-of-fold predictions are scored on
frames_depth.csv: AUC at 0.5-2 mm cutoffs and the per-approach metrics at P >= 0.8
(distance models: predicted <= 1 mm). Then, per depth seed, combined - image and
temporal - image AUC at 1 mm with a 95% bootstrap interval over approaches.

Usage: compare_runs.py [--work-dir ~/aliss_runs/contact_full]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from visual_proprioception.contact_train import (approach_metrics, auc,  # noqa: E402
                                                 causal_mean, depth_aucs, load_frames)

COLS = ['AUC0.5', 'AUC1', 'AUC1.5', 'AUC2', 'false alarm', 'before fire', 'by +1 s',
        'latency']


def load(work, run, model):
    path = os.path.join(work, run, model, 'oof_predictions.csv')
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        p = np.array([float(r['prob']) for r in csv.DictReader(f)])
    return -p if run.startswith('runR') else p   # distance models: higher score = closer


def row(score, thr, d, labels):
    a = depth_aucs(score, d)
    m = approach_metrics(score, d, thr, labels)
    return [a['0.5mm'], a['1mm'], a['1.5mm'], a['2mm'], m['false_alarm'], m['before_fire'],
            m['by_1s'], m['median_latency_s']]


def fmt(name, r):
    return (f'{name:30s}' + ''.join(f'{v:12.2f}' for v in r[:4])
            + ''.join(f'{v:12.0%}' for v in r[4:7]) + f'{r[7]:+11.2f}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('--work-dir', default=os.path.expanduser('~/aliss_runs/contact_full'))
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    work = args.work_dir

    d, meta = load_frames(os.path.join(work, 'frames_depth.csv'))
    labels = meta['labels']
    seeds = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(work, 'runD_seed*')))]
    print(f'{len(set(d["sequence"]))} approaches, {len(d["t"])} frames; depth labels; '
          f'thresholds: P >= 0.8 (yes/no models), predicted <= 1 mm (distance models)\n')
    print(f'{"run / model":30s}' + ''.join(f'{c:>12s}' for c in COLS))
    for model in ('spot', 'image', 'combined', 'temporal'):
        for run in ('runA_seed0', 'runR_seed0'):
            s = load(work, run, model)
            if s is not None:
                thr = -1.0 if run.startswith('runR') else 0.8
                print(fmt(f'{run[:4]} {model}', row(s, thr, d, labels)))
        scores = [s for s in (load(work, run, model) for run in seeds) if s is not None]
        if scores:
            r = np.array([row(s, 0.8, d, labels) for s in scores])
            lo, hi, mu = r.min(0), r.max(0), r.mean(0)
            print(f'{f"runD {model} ({len(scores)} seeds)":30s}'
                  + ''.join(f'{m:7.2f}±{(h - lo_) / 2:.2f}'
                            for m, lo_, h in zip(mu[:4], lo[:4], hi[:4]))
                  + ''.join(f'{m:8.0%}±{(h - lo_) / 2 * 100:2.0f}'
                            for m, lo_, h in zip(mu[4:7], lo[4:7], hi[4:7]))
                  + f'{mu[7]:+11.2f}s')
            print(fmt(f'runD {model} seed0 ~0.3s',
                      row(causal_mean(scores[0], d, 0.3), 0.8, d, labels)))
        print()

    # paired differences at 1 mm, bootstrap over approaches
    seqs = np.array(sorted(set(d['sequence'])))
    rows_of = {s: np.flatnonzero(d['sequence'] == s) for s in seqs}
    before = (d['t_rel'] < 0) & (d['laser'] != 1)
    pos = before & (d['depth_to_cut_mm'] <= 1.0) & (d['lateral_mm'] <= 2.0)
    neg = before & (d['depth_to_cut_mm'] >= 2.0)
    rng = np.random.default_rng(0)
    for run in seeds:
        base = load(work, run, 'image')
        for other in ('combined', 'temporal'):
            a = load(work, run, other)
            if a is None or base is None:
                continue
            diffs = []
            for _ in range(args.bootstrap):
                idx = np.concatenate([rows_of[s] for s in rng.choice(seqs, len(seqs))])
                diffs.append(auc(a[idx], pos[idx], neg[idx]) - auc(base[idx], pos[idx], neg[idx]))
            print(f'{run}: {other} - image AUC at 1 mm = '
                  f'{auc(a, pos, neg) - auc(base, pos, neg):+.3f} '
                  f'[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]')


if __name__ == '__main__':
    main()
