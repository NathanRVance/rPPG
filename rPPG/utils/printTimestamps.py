#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Print information about timestamps', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', nargs='+', help='csv formatted timestamps files')
    parser.add_argument('--merge-indices', nargs='+', type=int, help='Merge given indices with following value', default=[])
    parser.add_argument('--sum', action='store_true', help='Compute sum across timestamp files rather than average')
    parser.add_argument('--labels', nargs='+', help='Print labels rather than index numbers')
    parser.add_argument('--plot', help='Location to save a plot')
    parser.add_argument('--plotTitle', default='', help='Title when plotting')
    parser.add_argument('--fontFamilyOverride', help='Override the font family with, e.g., "serif"')
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    import csv

    timestamps = []
    for inp in args.input:
        with open(inp) as f:
            lines = list(csv.reader(f))
        for line in lines:
            line[0] = float(line[0])
        ts = [end[0]-start[0] for start, end in zip(lines, lines[1:])]
        mi = args.merge_indices[:]
        while len(mi) > 0:
            i = mi[0]
            ts[i] += ts[i+1]
            for j in range(i+1, len(ts)-1):
                ts[j] = ts[j+1]
            del ts[-1]
            del mi[0]
            mi = [i-1 for i in mi]
        # Convert to dict
        timestamps.append({i: t for i, t in enumerate(ts)})
    if args.sum:
        timestamps = [{i: sum(ts[i] for ts in timestamps) for i in timestamps[0].keys()}]
    if args.labels:
        timestamps = [{args.labels[i]: val for i, val in ts.items()} for ts in timestamps]
    # Output results
    if args.plot:
        import matplotlib.pyplot as plt
        if args.fontFamilyOverride:
            import matplotlib.font_manager
            plt.rcParams['font.family'] = [args.fontFamilyOverride]
        import numpy as np
        plt.figure()
        plt.title(args.plotTitle)
        plt.xlabel('Session Component')
        plt.ylabel('Duration')
        labels = list(timestamps[0].keys())
        avgs = {key: np.mean([ts[key] for ts in timestamps]) for key in labels}
        stddevs = {key: np.std([ts[key] for ts in timestamps]) for key in labels}
        plt.bar(labels, [avgs[key] for key in labels], yerr=[stddevs[key] for key in labels], capsize=5)
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.show()
    from rPPG.utils import tables
    if len(timestamps) > 1:
        timestamps = tables.mergeTables(timestamps, pm=' $\\pm$ ')
    else:
        timestamps = timestamps[0]
    tables.printTable(timestamps, delimiter=' & ', end=' \\\\', transpose=True)
