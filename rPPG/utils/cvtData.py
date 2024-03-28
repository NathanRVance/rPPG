#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import numpy as np
from rPPG.utils import npio
from rPPG.utils import metadata

def cvtFPS(data, inFPS, outFPS, resampleOverride=False):
    # Be smart enough to tell if it is a video, mask, or wave
    # if resampleOverride, then treats it as a wave (subject to signal.resample)
    data = np.array(data)
    if len(data.shape) == 1:
        # It is a wave
        if len(data) > 0 or resampleOverride:
            from scipy import signal
            return signal.resample(data, int(len(data) * outFPS / inFPS))
        else:
            return data
    elif len(data.shape) == 2:
        # It is a mask
        return [[int(bound * outFPS / inFPS) for bound in interval] for interval in data]
    else:
        # It is a video
        skip = round(inFPS / outFPS)
        return np.array(data[::skip])

def _cvtCSVs(args):
    fname, index, d, fps = args
    if len(d) == 0:
        return fname, index, []
    # Split into x and t for scipy.signal routine
    x, t = zip(*d)
    # Make t start at 0
    mint = min(t)
    t = [T - mint for T in t]
    if len(x) < 2:
        return fname, index, []
    else:
        from scipy import interpolate
        f = interpolate.interp1d(t, x, copy=False, assume_sorted=True)
        return fname, index, f(np.linspace(0, max(t), round(fps*max(t))))

def cvtJeremy(fname, fps=30):
    with np.load(fname) as data:
        t = data['wave_t']
        x = data['combined_wave']
        d = list(zip(x, t))
        return _cvtCSVs((None, None, d, fps))[-1]

def cvtCSVs(fnames, fps, waveformIndex=2):
    import csv
    data = {}
    for fname in fnames:
        data[fname] = []
        indices = [waveformIndex] if type(waveformIndex) is int else waveformIndex
        for index in indices:
            with open(fname) as f:
                # timestamp is -1
                data[fname].append([[float(line[index]), float(line[-1])] for line in csv.reader(f) if line[index]])
    # Make them all start at the same time
    maxMin = max(d[0][0][1] for d in data.values())
    data = {fname: [d[next((i for i in range(len(d)) if d[i][1] >= maxMin), len(d)-1):] for d in dat] for fname, dat in data.items()}
    from progress.bar import IncrementalBar
    from multiprocessing import Pool
    import os
    with Pool(int(os.getenv("NSLOTS", default = 0)) or None) as p:
        converted = [(fname, index, d) for fname, index, d in IncrementalBar('Converting CSVs', max=sum(len(dat) for dat in data.values())).iter(p.imap_unordered(_cvtCSVs, [(fname, index, d, fps) for fname, dat in data.items() for index, d in enumerate(dat)]))]
        ret = {parts[0]: {} for parts in converted}
        for fname, index, d in converted:
            ret[fname][index] = d
        return ret, maxMin

def cvtData(fnameIn, inputFPS=None, outputFPS=None):
    from pathlib import Path

    if Path(fnameIn).suffix == '.csv':
        if not outputFPS:
            print('ERROR: outputFPS required when processing a CSV input')
            exit(1)
        data = list(cvtCSVs([fnameIn], outputFPS, waveformIndex=[0,1,2])[0].values())[0]
        # We coopt inputFPS
        inputFPS = outputFPS
        meta = metadata.Metadata()
        meta.setFPS(outputFPS)
        meta['channels'] = {str(i): w for i, w in data.items()}
        data = meta['channels']['2']
    else:
        data, meta = npio.loadMeta(fnameIn)

    if inputFPS:
        inFPS = inputFPS
    elif meta:
        inFPS = meta.fps()
    else:
        print('ERROR: Input does not contain metadata nor is --inputFPS specified!')
        exit(1)

    outFPS = outputFPS if outputFPS else inFPS

    if outFPS != inFPS:
        inLen = len(data)
        data = cvtFPS(data, inFPS, outFPS)
        fpsReached = inFPS * len(data) / inLen
        print(f'Targeted {outFPS} FPS, and obtained {fpsReached} FPS')
        # Can be flat-out wrong sometimes
        #outFPS = fpsReached
    meta.setFPS(outFPS)
    
    return np.array(data), meta

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Convert npz/npy/hdf5/csv video or wave formats and sample rates', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='video or wave to process')
    parser.add_argument('output', help='output video or wave')
    parser.add_argument('--inputFPS', help='Override input FPS rather than use metadata', type=float)
    parser.add_argument('--outputFPS', help='Set the outputFPS rather than copy from inputFPS', type=float)
    parser.add_argument('--jeremy', action='store_true', help='Override as Jeremy\'s format')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.jeremy:
        data = cvtJeremy(args.input, args.outputFPS)
        meta = metadata.Metadata()
        meta.setFPS(args.outputFPS)
    else:
        data, meta = cvtData(args.input, args.inputFPS, args.outputFPS)
    npio.save(data, meta, args.output)
