"""
Merge Google Drive zip parts into one folder per session.

A Drive download of the resection export arrives as parts named
<session>-<stamp>Z-1-NNN/<session>/..., and one segment's frames are scattered across
all parts of its session. This links every file into <out_root>/<session>/... (hard
links: no extra space, originals untouched). A file present in several parts is linked
once if identical; differing copies are reported and the first is kept.

Usage: merge_parts.py <download_root> <out_root> [part_folder_substring ...]
"""
import argparse
from collections import Counter, defaultdict
import filecmp
import os
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('download_root')
    parser.add_argument('out_root')
    parser.add_argument('only', nargs='*', help='only part folders containing these strings')
    args = parser.parse_args()

    parts = defaultdict(list)
    for name in sorted(os.listdir(args.download_root)):
        m = re.match(r'(bladder_tele_.+?)-\d{8}T\d{6}Z-\d+-(\d+)$', name)
        if m and (not args.only or any(o in name for o in args.only)):
            parts[m.group(1)].append((m.group(2),
                                      os.path.join(args.download_root, name, m.group(1))))
    for session, plist in sorted(parts.items()):
        stats, conflicts = Counter(), []
        for _, base in plist:
            for dirpath, _, files in os.walk(base):
                rel = os.path.relpath(dirpath, base)
                os.makedirs(os.path.join(args.out_root, session, rel), exist_ok=True)
                for f in files:
                    a, b = os.path.join(dirpath, f), os.path.join(args.out_root, session, rel, f)
                    if not os.path.exists(b):
                        os.link(a, b)
                        stats['linked'] += 1
                    elif os.path.samefile(a, b) or filecmp.cmp(a, b, shallow=False):
                        stats['identical duplicate'] += 1
                    else:
                        conflicts.append(os.path.join(rel, f))
        print(f'{session}: parts {[p for p, _ in plist]} -> {dict(stats)}, '
              f'conflicts {len(conflicts)}')
        for c in conflicts[:10]:
            print('   conflict:', c)


if __name__ == '__main__':
    main()
