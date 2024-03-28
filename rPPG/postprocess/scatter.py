#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Scatter plot results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('bySubject', nargs='+', help='json-formatted by-subject evaluation to plot.')
parser.add_argument('output', help='Path to plot file to save')
parser.add_argument('--xMetric', default='SNR', help='Metric to plot on the x axis')
parser.add_argument('--yMetric', default='dhr', help='Metric to plot on the y axis')
parser.add_argument('--title', help='Override title for plot')
parser.add_argument('--highlight', nargs='+', help='Subject IDs to highlight')
parser.add_argument('--noerror', action='store_true', help='Suppress error bars, otherwise plots 95 percent CIs if multiple loss files are given.')
parser.add_argument('--ci', type=float, default=0.95, help='Confidence interval percent')
parser.add_argument('--printSorted', choices=('x', 'y'), default=None, help='Print averages, sorted by either the x or y metric')

argcomplete.autocomplete(parser)
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
from rPPG.utils.tables import ci

plt.figure()
if args.title:
    plt.title(args.title)
else:
    plt.title(f'{args.xMetric} vs {args.yMetric}')
plt.xlabel(args.xMetric)
plt.ylabel(args.yMetric)

def scatterPlot(data, label=None):
    # Data is [{metric: [values]}, ... ]
    x = [d[args.xMetric] for d in data]
    y = [d[args.yMetric] for d in data]
    x, y = [[(np.mean(vals), ci(vals, alpha=args.ci)) for vals in series] for series in (x, y)]
    x, xErr = zip(*x)
    y, yErr = zip(*y)
    if args.noerror:
        plt.scatter(x, y, label=label)
    else:
        plt.errorbar(x, y, xerr=xErr, yerr=yErr, fmt='o', label=label)

import json
bySubject = {}
for fname in args.bySubject:
    with open(fname) as f:
        for subID, metrics in json.load(f).items():
            if subID not in bySubject:
                bySubject[subID] = {}
            for metric, value in metrics.items():
                if metric not in bySubject[subID]:
                    bySubject[subID][metric] = []
                bySubject[subID][metric].append(value)

scatterPlot(bySubject.values())

if args.highlight:
    for subID in args.highlight:
        scatterPlot([bySubject[subID]], label=subID)
    plt.legend()

if args.printSorted:
    # Also print extrema
    metric = args.xMetric
    if args.printSorted == 'y':
        metric = args.yMetric
    print(f'Sorted by {metric}:')
    for subID in sorted(bySubject.keys(), key=lambda sid: np.mean(bySubject[sid][metric])):
        print(f'{subID}: {bySubject[subID][metric]}')


plt.savefig(args.output)
plt.show()
