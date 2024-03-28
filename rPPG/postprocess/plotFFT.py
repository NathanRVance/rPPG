#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Plot FFT windows', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('fft', help='fft data to analyze')
parser.add_argument('output', help='Path to save plot')
parser.add_argument('--index', type=int, help='Index in the fft data to plot')
parser.add_argument('--title', help='Title for the plot', default='FFT Spectrogram')
parser.add_argument('--ylabel', help='Override Y label for the plot')
parser.add_argument('--xlabel', help='Override X label for the plot')
parser.add_argument('--overlay', help='Wave or HR file to overlay the fft')

argcomplete.autocomplete(parser)
args = parser.parse_args()

import matplotlib.pyplot as plt
from rPPG.utils import npio
from scipy import signal
import numpy as np

plt.figure()
plt.title(args.title)

fft, meta = npio.loadMeta(args.fft)

if args.index == None:
    # fft might have scale embedded, detect based on shape
    if fft.shape[1] == 2:
        # Just throw it away for now
        fft = fft[:, 1, :]
        # We're also inverted from what's expected below
        fft = np.swapaxes(fft, 0, 1)
        fft = np.flip(fft, 0)
    import numpy as np
    plt.xlabel('Time (s)')
    plt.ylabel('HR (BPM)')
    plt.imshow(np.abs(fft), aspect='auto', cmap='turbo', extent=[0, fft.shape[1]/meta.fps(), 40, 180])
    #plt.imshow(np.abs(fft), aspect='auto', cmap='turbo')
else:
    print(f'Taking index {args.index} in fft (range 0 - {len(fft)})')
    plt.xlabel('Density')
    plt.ylabel('Frequency (hz)')
    x, y = fft[args.index]
    plt.plot(x, y, label=args.fft)
    print('Dominant peaks:')
    for i in signal.find_peaks(y)[0]:
        print(f'Frequency {x[i]:.4f}: Density {y[i]:.4f}')

if args.overlay:
    data, meta = npio.loadMeta(args.overlay)
    x = [i/meta.fps() for i in range(len(data))]
    plt.plot(x, data, label=args.overlay, linestyle='--', color='k')

if args.xlabel:
    plt.xlabel(args.xlabel)
if args.ylabel:
    plt.ylabel(args.ylabel)
plt.savefig(args.output)
plt.show()
