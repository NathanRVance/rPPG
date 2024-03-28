#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Create mask from timestamps', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='csv formatted timestamps file')
    parser.add_argument('indices', nargs='+', help='Indices in input to make masked (zero-indexed, of course!)', type=int)
    parser.add_argument('output', help='output file')
    parser.add_argument('--skip-normalize', action='store_true', help='Skip normalizing such that starting timestamp is zero')
    parser.add_argument('--fps', type=float, default=30, help='Framerate when converting to frame numbers')
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    import csv
    with open(args.input) as f:
        lines = list(csv.reader(f))
    for line in lines:
        line[0] = int(float(line[0]) * args.fps)
    if not args.skip_normalize:
        start = lines[0][0]
        for line in lines:
            line[0] -= start
    mask = []
    for i in args.indices:
        print(f'Making mask starting at {lines[i][1]}')
        mask.append([lines[i][0], lines[i+1][0]])
    from rPPG.utils import npio
    npio.save(mask, {}, args.output)
