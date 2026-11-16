"""
Break one run's held-out predictions down by session, approach and time before firing.

For <run_dir>/<model>/oof_predictions.csv (default runD_seed0/combined, the shipped
model) scored on frames_depth.csv with the 1 mm / 2 mm depth labels:
- AUC at 1 mm and median P near / far for each held-out session
- within-approach AUC (near and far frames from the same approach only)
- median P near / far by time before laser-on

Usage: per_session.py [--work-dir ~/aliss_runs/contact_full] [--run runD_seed0]
                      [--model combined]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from visual_proprioception.contact_train import auc, load_frames  # noqa: E402


def median(x):
    return float(np.median(x)) if len(x) else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('--work-dir', default=os.path.expanduser('~/aliss_runs/contact_full'))
    parser.add_argument('--run', default='runD_seed0')
    parser.add_argument('--model', default='combined')
    args = parser.parse_args()

    d, _ = load_frames(os.path.join(args.work_dir, 'frames_depth.csv'))
    path = os.path.join(args.work_dir, args.run, args.model, 'oof_predictions.csv')
    with open(path, newline='') as f:
        p = np.array([float(r['prob']) for r in csv.DictReader(f)])
    before = (d['t_rel'] < 0) & (d['laser'] != 1)
    pos = before & (d['depth_to_cut_mm'] <= 1.0) & (d['lateral_mm'] <= 2.0)
    neg = before & (d['depth_to_cut_mm'] >= 2.0)
    print(f'{args.run}/{args.model}: AUC at 1 mm {auc(p, pos, neg):.3f}\n')

    print('per held-out session:')
    sess = np.array(d['session'])
    for s in sorted(set(sess)):
        m = sess == s
        print(f'  {s:36s} AUC1 {auc(p[m], pos[m], neg[m]):.3f}   near {pos[m].sum():5d} '
              f'far {neg[m].sum():6d}   median P near {median(p[m & pos]):.2f} '
              f'far {median(p[m & neg]):.2f}')

    seq = np.array(d['sequence'])
    w, ws = [], []
    for q in set(seq[pos]) & set(seq[neg]):
        m = seq == q
        w.append(auc(p[m], pos[m], neg[m]))
        ws.append(pos[m].sum() * neg[m].sum())
    print(f'\nwithin-approach AUC1 (near/far pairs from the same approach only): '
          f'{np.average(w, weights=ws):.3f} over {len(w)} approaches')

    print('\nby time before laser-on:')
    rel = d['t_rel']
    for lo, hi in ((-0.5, 0), (-2, -0.5), (-5, -2), (-60, -5)):
        m = (rel >= lo) & (rel < hi)
        print(f'  t_rel in [{lo}, {hi}) s: near frames {int((pos & m).sum()):6d} median P '
              f'{median(p[pos & m]):.2f}; far frames {int((neg & m).sum()):6d} median P '
              f'{median(p[neg & m]):.2f}')


if __name__ == '__main__':
    main()
