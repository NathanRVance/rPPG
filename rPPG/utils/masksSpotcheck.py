#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from rPPG.utils import npio
from rPPG.utils import hr
from rPPG.utils import masks
from rPPG.utils import combineWaves

def plotWaveHR(wave, plotTitle):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(plotTitle)
    plt.xlabel('Time (s)')
    plt.ylabel('Waveform and HR (BPM)')
    # Scale wave to same mean and stddev as HR
    hr = wave['hr']
    def normalize(data):
        data *= np.std(hr) / np.std(data)
        return data + np.mean(hr) - np.mean(data)
    dat = wave.data()
    cms = wave['components']['cms50ea']
    dat = dat[:min(len(dat), len(cms))]
    cms = cms[:min(len(dat), len(cms))]
    timeDiff = int((len(dat)-len(hr))//2)
    times = [float(i)/wave.fps() for i in range(len(dat))][timeDiff:]
    dat = normalize(dat[timeDiff:])
    cms = normalize(cms[timeDiff:])
    # Plot everything
    plt.plot(times, dat, color='blue')
    plt.plot(times, cms, color='blue', linewidth=0.5)
    plt.plot(times[:len(hr)], hr, color='red')
    plt.plot(times[:len(hr)], masks.applyMaskFill(hr, wave.mask()), color='green')
    plt.show()

def solicitMaskEdits(mask):
    import tempfile
    import json
    import subprocess, os
    maskFile = tempfile.NamedTemporaryFile('w', delete=False)
    maskFile.write(json.dumps(mask, indent=2))
    maskFile.close()
    subprocess.run(['vi', maskFile.name])
    with open(maskFile.name, 'r') as f:
        mask = json.load(f)
    os.unlink(maskFile.name)
    return mask

def displayAndEdit(wave, plotTitle):
    import multiprocessing
    multiprocessing.Process(target=plotWaveHR, args=(wave, plotTitle)).start()
    return solicitMaskEdits(wave.mask())

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Spotcheck wave masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('wave', help='npz formatted waves file to spotcheck')
    parser.add_argument('output', help='Path to save spotchecked output')
    parser.add_argument('--hr', help='Use precalculated HR')
    parser.add_argument('--window', type=float, default=10, help='Window size in seconds')
    parser.add_argument('--bandpass', action='store_true', help='Apply a bandpass filter to the wave')
    #parser.add_argument('--raw', nargs='+', help='Include raw wave')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    wave = npio.meta(args.wave)
    if args.bandpass:
        wave.setData(combineWaves.bandpass(wave.data(), wave.fps()))
    if args.hr:
        wave['hr'] = npio.load(args.hr)
    if 'hr' not in wave:
        wave['hr'] = hr.calcHRFFT(wave.data(), wave.fps(), args.window)[0]
    #plotWaveHR(wave, HR, args.wave)
    #print(f'Edited: {solicitMaskEdits(wave.mask())}')
    print(f'Edited: {displayAndEdit(wave, args.wave)}')
