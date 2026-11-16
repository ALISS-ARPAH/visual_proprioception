"""
Check that a merged data root is complete before building the frame table.

Layout: <root>/<session>/rosbags/lesion_*/cut_*_rect_parsed. Per session it counts cuts
with labels.json, segments with kinematics.csv / annotation.json / endoscope frames, and
frames present vs frames listed in each kinematics.csv. Optionally it also checks every
frame listed in the colleague's stitched_manifest.jsonl (from BladderTouchDynamicCrop).

Usage: verify_data.py <root> [stitched_manifest.jsonl]
"""
import argparse
from collections import Counter
import csv
import glob
import json
import os


def check_session(root, session):
    cuts = sorted(glob.glob(os.path.join(root, session, 'rosbags', 'lesion_*',
                                         'cut_*_rect_parsed')))
    c = Counter()
    for cut in cuts:
        c['cuts'] += 1
        c['labels.json'] += os.path.exists(os.path.join(cut, 'labels.json'))
        for seg in glob.glob(os.path.join(cut, 'resect*', '*')):
            c['segments'] += 1
            kin = os.path.join(seg, 'kinematics.csv')
            c['annotation'] += os.path.exists(os.path.join(seg, 'annotation.json'))
            endoscope = os.path.join(seg, 'endoscope')
            imgs = set(os.listdir(endoscope)) if os.path.isdir(endoscope) else set()
            if os.path.exists(kin):
                c['kinematics'] += 1
                with open(kin, newline='') as f:
                    names = {r['filename'] for r in csv.DictReader(f)}
                c['frames listed'] += len(names)
                c['frames present'] += len(names & imgs)
                c['segments missing frames'] += bool(names - imgs)
    ok = (c['labels.json'] == c['cuts'] and c['kinematics'] == c['segments']
          and c['frames present'] == c['frames listed'])
    return ok, c


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('root')
    parser.add_argument('manifest', nargs='?', help="colleague's stitched_manifest.jsonl")
    args = parser.parse_args()

    ok_all = True
    for session in sorted(os.listdir(args.root)):
        ok, c = check_session(args.root, session)
        ok_all &= ok
        print(f'{"OK  " if ok else "FAIL"} {session}: {dict(c)}')
    if args.manifest:
        want, have = Counter(), Counter()
        with open(args.manifest) as f:
            for line in f:
                e = json.loads(line)
                s = e['session']
                local = next((d for d in os.listdir(args.root) if d.lower() == s.lower()), s)
                for frame in e['frames']:
                    want[s] += 1
                    have[s] += os.path.exists(
                        os.path.join(args.root, local, frame.split(s + '/', 1)[1]))
        for s in sorted(want):
            print(f'     colleague manifest {s}: {have[s]}/{want[s]} frames present')
            ok_all &= have[s] == want[s]
    print('ALL COMPLETE' if ok_all else 'INCOMPLETE')


if __name__ == '__main__':
    main()
