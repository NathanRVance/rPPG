#!/usr/bin/env python3

def mergeMasks(masks):
    import numpy as np
    out = []
    for interval in sorted(np.concatenate(masks), key=lambda x: x[0]):
        if len(out) == 0 or out[-1][1] < interval[0]:
            out.append(interval)
        else:
            out[-1][1] = max(out[-1][1], interval[1])
    return out

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Merge two or more masks together', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='npz formatted mask to merge')
    parser.add_argument('merge', nargs='+', help='additional masks to merge')
    parser.add_argument('output', help='output file')
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from rPPG.utils import npio
    mask, meta = npio.loadMeta(args.input)
    masks = [mask] + [npio.load(inputPath) for inputPath in args.merge]
    npio.save(mergeMasks(masks), meta, args.output)
    
