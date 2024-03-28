#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Perform spectral analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('fft', help='fft data to analyze')
parser.add_argument('--gaussCount', type=int, help='Number of gaussians to use', default=3)
parser.add_argument('--saveParams', help='Path to save gaussian parameters')
parser.add_argument('--loadParams', help='Load gaussian parameters rather than recalculate')
parser.add_argument('--saveVideo', help='Path to save AVI video output')
parser.add_argument('--saveHR', help='Path to save npz formatted heart rate')
parser.add_argument('--hrMethod', choices=['simple', 'optimize'], help='Method to use for HR calculation', default='simple')
parser.add_argument('--moveCost', type=float, help='Cost to change HR when --hrMethod=optimize', default=10)

argcomplete.autocomplete(parser)
args = parser.parse_args()

import cv2
import numpy as np
from scipy import signal
from rPPG.utils import npio
from rPPG.utils import hr
from progress.bar import IncrementalBar

def gaussAUC(A, sig):
    return A * sig * np.sqrt(2*np.pi)
def gaussian(x, A, x0, sig, offset=0):
    return offset + A*np.exp(-(x-x0)**2/(2*sig**2))
def multiGaussian(x, *pars):
    return sum(gaussian(x, A, x0, sig) for A, x0, sig in zip(pars[::3], pars[1::3], pars[2::3]))

def plotGaussFFT(x, y, title, gaussParams):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('frequency')
    plt.ylabel('density')
    plt.plot(x, y, label='fft')
    plt.plot(x, multiGaussian(x, *gaussParams), label='MultiGauss')
    for A, x0, sig in zip(gaussParams[::3], gaussParams[1::3], gaussParams[2::3]):
        plt.plot(x, gaussian(x, A, x0, sig), label='Gauss')
    plt.legend()
    plt.tight_layout()
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close('all')
    return img

def fitMultiGauss(x, y, n):
    """Fit n gaussians to x and y

    Returns:
      Array with [Amplitude, Mean, stddev] for each gaussian for each n
    """
    from scipy.optimize import curve_fit
    N = n
    for n in range(N, 0, -1):
        peaksAll = signal.find_peaks(y)[0]
        peaks = sorted(sorted(peaksAll, key=lambda i: y[i])[-n:])
        sig = (max(x)-min(x))/n # starting guess for component width
        guess = np.ravel([[y[i], x[i], sig] for i in peaks]) # starting guess for (amp, x, width) for each component
        def getAdjacents(dist, val):
            lowers = [d for d in dist if d < val]
            lower = 0 if not lowers else max(lowers)
            uppers = [d for d in dist if d > val]
            upper = len(x)-1 if not uppers else min(uppers)
            return lower, upper
        lbound = sum(([0, (x[i]+x[getAdjacents(peaksAll, i)[0]])/2, 0] for i in peaks), [])
        ubound = sum(([2*y[i], (x[i]+x[getAdjacents(peaksAll, i)[1]])/2, 2*sig] for i in peaks), [])
        try:
            result = curve_fit(multiGaussian, x, y, guess, bounds=(lbound, ubound))
        except RuntimeError:
            # Hit max iterations...
            continue
        #print(f'Guess: {guess}')
        #print(f'Bounds: {(lbound, ubound)}')
        #print(f'Result: {result[0]}')
        return result[0]
    print('Uh oh, we shouldn\'t get here!')

def calcHRSimple(gaussParams):
    HR = []
    for params in gaussParams:
        _, maxFreq, _ = max([Ax0sig for Ax0sig in zip(params[::3], params[1::3], params[2::3])], key=lambda x: gaussAUC(x[0], x[2]))
        HR.append(maxFreq * 60)
    return HR

def calcHRSSSP(fft, gaussParams, moveCost):
    scores = []
    for (x, y), params in zip(fft, gaussParams):
        s = [(x0, gaussAUC(A, sig)) for A, x0, sig in zip(params[::3], params[1::3], params[2::3])]
        s += [(x[i], y[i]) for i in signal.find_peaks(y)[0]]
        scores.append(list(reversed(sorted(s, key=lambda sc: sc[1]))))
    # Find path that maximizes score and minimizes movement: single source shortest/longest path in DAG
    # Maximize edge weight = score - moveCost * distance
    maxScores = [(s[0], 0, None) for s in scores[0]] # reflects scores array; is (x0, score, parent)
    for sc in scores[1:]:
        #print(f'Processing maxScores: {maxScores}')
        #print(f'Processing prev: {prev}')
        #print(f'Processing cur: {cur}')
        maxScoresNext = []
        for x0, score in sc:
            best = max(maxScores, key=lambda ms: ms[1]+score-(moveCost*abs(x0-ms[0])))
            maxScoresNext.append((x0, best[1]+score-(moveCost*abs(x0-best[0])), best))
        maxScores = maxScoresNext
    maxScore = max(maxScores, key=lambda ms: ms[1])
    HR = []
    while maxScore is not None:
        HR.append(maxScore[0])
        maxScore = maxScore[2]
    HR = [x * 60 for x in reversed(HR)]
    return HR

fft, meta = npio.loadMeta(args.fft)
fps = meta['streams'][0]['avg_frame_rate']

if args.loadParams:
    params = npio.load(args.loadParams)
else:
    params = [fitMultiGauss(x, y, args.gaussCount) for i, (x, y) in IncrementalBar('Fitting Gaussians', max=len(fft), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(enumerate(fft))]

if args.saveParams:
    npio.save(params, meta, args.saveParams)

if args.saveVideo:
    print('Writing video')
    out = None
    for i, ((x, y), params) in IncrementalBar('Plotting Gaussians', max=len(fft), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(enumerate(zip(fft, params))):
        frame = plotGaussFFT(x, y, f'Time: {i/fps:.4f} seconds', params)
        if out is None:
            h, w, _ = frame.shape
            out = cv2.VideoWriter(args.saveVideo, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(w), int(h)))
        out.write(frame)

if args.saveHR:
    if args.hrMethod == 'simple':
        HR = calcHRSimple(params)
    elif args.hrMethod == 'optimize':
        HR = calcHRSSSP(fft, params, args.moveCost)
    else:
        raise ValueError(f'Unknown hr calculation method: {args.hrMethod}')
    npio.save(HR, meta, args.saveHR)
