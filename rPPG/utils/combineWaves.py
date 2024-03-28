#!/usr/bin/env python3
import scipy
import numpy as np
from pathlib import Path
from rPPG.utils import npio
from rPPG.utils import hr
from rPPG.utils import evaluate
from rPPG.utils import masks
from rPPG.utils import metadata

def bandpass(wave, fps, cuttoffs=[40/60, 180/60], order=4):
    b, a = scipy.signal.butter(order, Wn=cuttoffs, btype='bandpass', fs=fps)
    return scipy.signal.filtfilt(b, a, wave, axis=0)

def calcMAE(data, window):
    combined = combine(data)
    wavesDict = {wave.location(): wave.data() for wave in data}
    wavesDict['combined'] = combined.data()
    #HRs = {fname: HR for fname, (HR, fft) in hr.calcHR(wavesDict, data[0].fps(), window, skipFFT=False).items()}
    HRs = hr.calcHR(wavesDict, data[0].fps(), window, skipFFT=False)
    FFTs = {fname: fft for fname, (HR, fft) in HRs.items()}
    HRs = {fname: HR for fname, (HR, fft) in HRs.items()}
    del wavesDict['combined']
    combinedHR = HRs['combined']
    del HRs['combined']
    combinedFFT = FFTs['combined']
    del FFTs['combined']
    # Calc MAE
    MAE = np.mean([np.mean(np.abs(combinedHR[:len(indiv)]-indiv[:len(combinedHR)])) for indiv in HRs.values()])
    return MAE, HRs, combined, combinedHR, FFTs, combinedFFT

def _genMasks(args):
    wave, HR, FFT, window, method = args
    snrs = None
    if method == 'snr':
        snrs = evaluate.snr(FFT, HR, returnAvg=False)
        # Save version of snrs padded to be same length as wave
        wave['snr'] = [snrs[0]] * ((len(wave.data())-len(snrs))//2) + snrs
        wave['snr'] = wave['snr'] + [wave['snr'][-1]] * (len(wave.data()-len(wave['snr'])))
        # Finally, undo the snr = 10*log10(signal/noise) operation
        wave['snr'] = [10**(s/10) for s in wave['snr']]
    wave.setMask(masks.calcMask(wave.data(), wave.fps(), HR=HR, fft=FFT, window=window, snrs=snrs, criterion=method)[0])
    return wave

def genMasks(data, combined, HRs, combinedHR, FFTs, window=10, method='dhr'):
    keyOrder = [d.location() for d in data]
    from progress.bar import IncrementalBar
    from multiprocessing import Pool
    import os
    with Pool(int(os.getenv("NSLOTS", default = 0)) or None) as p:
        masked = {d.location(): d for d in IncrementalBar('Calculating Masks', max=len(data), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(p.imap_unordered(_genMasks, [(wave, HRs[wave.location()], FFTs[wave.location()], window, method) for wave in data]))}
    return [masked[location] for location in keyOrder]

def combineAvg(waves):
    combined = metadata.Metadata()
    if 'snr' in waves[0]:
        # Take weighted average by snr (calculated for each point)
        combined.setData(np.array([sum(w.data()[i]*w['snr'][i] for w in waves if min(len(w.data()), len(w['snr'])) > i)/sum(w['snr'][i] for w in waves if len(w['snr']) > i) for i in range(max(len(wave.data()) for wave in waves))]))
    else:
        # Each point in combined is an average of all unmasked points in the source
        combined.setData(np.array([np.mean([w.data()[i] for w in waves if len(w.data()) > i and not any(i > start and i < end for start, end in w.mask())]) for i in range(len(waves[0].data()))]))
    #ws = [w.data() for w in waves]
    #combined.setData(np.mean(ws, axis=0))
    combined.setFPS(waves[0].fps())
    combined.setLocation('combined')
    return combined

def shift(combined, toShift):
    #lags = scipy.signal.correlation_lags(len(combined.data()), len(toShift.data()))
    lags = np.arange(-len(toShift.data())+1, len(combined.data()))
    corr = scipy.signal.correlate(combined.data(), toShift.data())
    # Shift by up to a second
    maxShift = int(toShift.fps())
    maxloc = max((i for i in range(len(corr)//2-maxShift, len(corr)//2+maxShift)), key=lambda i: abs(corr[i]))
    lag = lags[maxloc]
    #print(f'Lag for {toShift.location()}: {lag}')
    sh = toShift.data()
    if lag < 0:
        sh = np.concatenate((sh[abs(lag):], [0]*abs(lag)))
    elif lag > 0:
        sh = np.concatenate(([0]*abs(lag), sh))
    # if equal, do nothing
    if corr[maxloc] < 0:
        sh *= -1
    shObj = metadata.Metadata()
    if toShift.mask() is not None: # Shift the mask
        shObj.setMask([[p1+lag, p2+lag] for p1, p2 in toShift.mask()])
    shObj.setData(sh)
    if 'snr' in toShift:
        shObj['snr'] = toShift['snr']
    shObj.setFPS(toShift.fps())
    shObj.setLocation(toShift.location())
    return shObj

def combine(waves):
    # Combine by taxing MXCorr (with possible invert) and averaging
    # Use scipy.signal.correlate
    # How to handle n-way correlation? First combine them all naively, then calc mxcorr based on that.
    combined = combineAvg(waves)
    # Find max abs xcorr for each
    shifted = [shift(combined, wave) for wave in waves]
    combined = combineAvg(shifted)
    combined['components'] = {}
    for wave in waves:
        combined['components'][wave.location()] = wave.data()
    return combined

def plot(waves, HRs, combinedWave, combinedHR, MAE, outdir, methodName):
    import matplotlib.pyplot as plt
    fps = waves[0].fps()
    locations = [w.location() for w in waves]
    masks = [w.mask() for w in waves]
    waves = [w.data() for w in waves]
    HRs = [HRs[loc] for loc in locations]
    hrOffset = (len(waves[0]) - len(HRs[0])) // 2
    hrMasks = [[[p1-hrOffset, p2-hrOffset] for p1, p2 in mask] for mask in masks]
    combinedHRMask = [[p1-hrOffset, p2-hrOffset] for p1, p2 in combinedWave.mask()]
    def makePlot(name, ylabel, data, dataLabels, combined, start, end, masks, combinedMask=None):
        plt.figure(figsize=(12,6))
        plt.title(f'{name} with MAE={MAE:.3f}; method: {methodName}')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        ax = plt.gca()
        times = [float(i) / fps for i in range(len(combined))]
        for d, label, mask in zip(data, dataLabels, masks):
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(times[start:min(end, len(d))], d[start:end], label=f'{label} (masked)', linewidth=.5, color=color)
            masked = [np.nan if any(i > interval[0] and i < interval[1] for interval in mask) else val for i, val in enumerate(d)]
            plt.plot(times[start:min(end, len(masked))], masked[start:end], label=label, linewidth=1.5, color=color)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times[start:end], combined[start:end], label='combined (masked)', linewidth=.5, color=color)
        if combinedMask:
            combined = [np.nan if any(i > interval[0] and i < interval[1] for interval in combinedMask) else val for i, val in enumerate(combined)]
        plt.plot(times[start:end], combined[start:end], label='combined', linewidth=3, color=color)
        plt.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outdir / f'{name}.svg')
        #plt.show()
        #plt.close()
        return plt
    makePlot('Waves', 'Waveform', waves, locations, combinedWave.data(), int(10*fps), int(20*fps), masks, combinedWave.mask())
    return makePlot('HRs', 'BPM', HRs, locations, combinedHR, 0, len(combinedHR), hrMasks, combinedHRMask)

def combineWaves(waves: list, method: str = 'snr', window: float = 10, plotdir: str = None) -> metadata.Metadata:
    ''' Combine waves into singe metadata object

    Arguments:
      waves: list of metadata.Metadata objects
      method: Method used to combine waves, either snr (default) or dhr
      window: Window size in seconds used for fft
      plotdir: Directory to save plots (if provided)
    
    Returns:
      Combined wave as a metadata.Metadata object
    '''
    # Bandpass them
    for wave in waves:
        wave.setData(bandpass(wave.data(), wave.fps()))
        wave.setMask([])
    waves = [w for w in waves if len(w.data()) > 0]
    MAE, HRs, combined, combinedHR, FFTs, combinedFFT = calcMAE(waves, window)
    waves = genMasks(waves, combined, HRs, combinedHR, FFTs, window=window, method=method)
    MAE, HRs, combined, combinedHR, FFTs, combinedFFT = calcMAE(waves, window)
    # Also calc dhr masks for combined just for kicks
    combined.setMask(masks.calcMask(combined.data(), combined.fps(), combinedHR, combinedFFT, window=window, criterion='dhr')[0])
    if plotdir:
        plotdir = Path(plotdir)
        plotdir.mkdir(parents=True, exist_ok=True)
        plot(waves, HRs, combined, combinedHR, MAE, plotdir, method)
    return combine(waves)

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Combine two or more waves into one', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('waves', nargs='+', help='CSV (or npz) formatted waves to combine')
    parser.add_argument('outdir', help='Directory to write output')
    parser.add_argument('--fps', type=float, default=90, help='Frames per second')
    parser.add_argument('--window', type=float, default=10, help='Window size in seconds')
    parser.add_argument('--method', default='snr', choices=('dhr', 'snr', 'majorityVote'), help='Method for generating masks')
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
  
    from rPPG.utils import cvtData
    if Path(args.waves[0]).suffix == '.csv':
        data = {fname: metadata.Metadata() for fname in args.waves}
        for p, d in data.items():
            d.setFPS(args.fps)
            d.setLocation(Path(p).stem.split('_')[-1])
        converted, timeStart = cvtData.cvtCSVs(args.waves, args.fps, waveformIndex=[0,1,2])
        for fname, channels in converted.items():
            d = data[fname]
            d.setStartTime(timeStart)
            for i, w in channels.items():
                if 'channels' not in d:
                    d['channels'] = {}
                d['channels'][str(i)] = w
                if (str(i) == '2' and d.location() == 'cms50ea') or (str(i) == '1' and d.location() != 'cms50ea'):
                    d.setData(w)
        # Convert data to list
        data = list(data.values())
    else:
        data = [npio.meta(fname) for fname in args.waves]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combined = combineWaves(data, method=args.method, window=args.window, plotdir=args.outdir)
    npio.save(combined.data(), combined, outdir / (combined.location() + '.npz'))
