#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete
import numpy as np

parser = argparse.ArgumentParser(description='Plot the waveform over time', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('waveform', nargs='+', help='npz formatted waveforms')
parser.add_argument('output', help='Path to save plot')
parser.add_argument('--fps', help='Override frames per second of video and waveform as recorded in wave metadata', default=None, type=float)
parser.add_argument('--start', help='x min, in seconds', default=0, type=float)
parser.add_argument('--end', help='x max, in seconds', default=-1, type=float)
parser.add_argument('--normalize', help='Normalize waveform to [-1, 1]', action='store_true')
parser.add_argument('--center', action='store_true', help='If waveform lengths differ, center on each other when plotting')
parser.add_argument('--title', help='Title for the plot', default='Waveform')
parser.add_argument('--ylabel', help='Y label for the plot', default='Waveform')
parser.add_argument('--xlabel', help='X label for the plot', default='Time (s)')
parser.add_argument('--legendIndex', type=int, help='Position in wave path used for the legend. Default is filename stem. Reverse-indexing supported via negative values.')
parser.add_argument('--legendOverride', nargs='+', help='Override labels by explicitely setting values')
parser.add_argument('--legendOutside', action='store_true', help='Position the legend outside of the plot')
parser.add_argument('--fontFamilyOverride', help='Override the font family with, e.g., "serif"')
parser.add_argument('--figsize', type=float, nargs=2, default=[6.4, 4.8], help='Plot dimensions')

argcomplete.autocomplete(parser)
args = parser.parse_args()

import matplotlib.pyplot as plt
from rPPG.utils import npio
from pathlib import Path
import numpy as np
if args.fontFamilyOverride:
    import matplotlib.font_manager
    plt.rcParams['font.family'] = [args.fontFamilyOverride]

plt.figure(figsize=args.figsize)
plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)

waves = {fname: list(npio.loadMeta(fname)) for fname in args.waveform}
if args.center:
    # Pad shorter waves to be centered on longest
    maxLen = max(len(wavemeta[0]) for wavemeta in waves.values())
    for key in list(waves.keys()):
        if maxLen > len(waves[key][0]):
            waves[key][0] = [np.nan] * ((maxLen - len(waves[key][0]))//2) + waves[key][0].tolist()

for i, (fname, (wave, metadata)) in enumerate(waves.items()):
    if not args.fps:
        if metadata:
           args.fps = metadata.fps()
        else:
            print('ERROR: No --fps provided and no metadata in wave file')
            exit(1)
    times = [float(i) / args.fps for i in range(len(wave))]
    start = int(args.start * args.fps)
    end = int(args.end * args.fps)
    if end > 0:
        wave = wave[:end]
        times = times[:end]
    wave = wave[start:]
    times = times[start:]
    if args.normalize:
        wave = np.array(wave) *  2/(max(wave) - min(wave))
        wave -= min(wave) + 1
    if args.legendOverride:
        label = args.legendOverride[i]
    else:
        label = Path(fname).stem if args.legendIndex == None else Path(fname).parts[args.legendIndex]
    plt.plot(times, wave, label=label)

if len(args.waveform) > 1:
    if args.legendOutside:
        plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    else:
        plt.legend()
plt.tight_layout()
plt.savefig(args.output)
plt.show()
