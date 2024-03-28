#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import numpy as np
from multiprocessing import Pool
from progress.bar import IncrementalBar
from pathlib import Path
import json
from rPPG.utils import masks as msk
from rPPG.utils import npio
from rPPG.utils import hr
from rPPG.utils import envelope


def getCorrs(static: list, dynamic: list, shiftSearch: tuple) -> list:
    """Calculate pearson r correlations between static and dynamic
    
    Shifting means moving dynamic by shift relative to static:
    Positive slides dynamic to the right, which is the same as taking
    static[shift:]
    or equivalently:
    dynamic = [0]*shift + dynamic

    Arguments:
      static: The wave that is held static
      dynamic: The wave that is shifted
      shiftSearch: 2-length tuple of bounds to consider for shifting

    Returns:
      list: A list of [(shift, r)] tuples indicating the amount that dynamic was shifted
    """
    from scipy.stats import pearsonr
    search = list(range(int(shiftSearch[0]), int(shiftSearch[1])))
    if not search:
        return 0
    def getPearsonr(s, d, shift):
        if shift < 0:
            d = d[-shift:]
        else:
            s = s[shift:]
        minLen = min(len(s), len(d))
        s = s[:minLen]
        d = d[:minLen]
        return pearsonr(s, d)[0]
    return [(s, getPearsonr(static, dynamic, s)) for s in search]

def getShift(static: list, dynamic: list, maxShiftFrames: int) -> int:
    """Return the number of frames to shift the dynamic wave to optimize pearsonr

    Arguments:
      static: The wave that is held static
      dynamic: The wave that is shifted
      maxShiftFrames: Number of frames to consider when shifting

    Returns:
      int: The number of frames to shift the dynamic wave. See docs for getCorrs for details.
    """
    return max(getCorrs(static, dynamic, (-maxShiftFrames, maxShiftFrames)), key=lambda c: c[1])[0]

def dhr(hr, fps, dlim=7):
    return sum(max(0, fps*abs(a-b)-dlim) for a, b in zip(hr, hr[1:])) / (len(hr)-1)

def snr(fft: list, gtHR: list, allowance: float = 2.0, returnAvg=True) -> float:
    '''Calculate the Signal to Noise Ratio

    Arguments:
      fft: [[freq, density], ...] formatted fft data
      gtHR: Ground truth heart rate at each frame
      allowance: BPM allowance around the peak that qualifies as "signal"
      returnAvg: Average results before returning, otherwise is snr for each timepoint

    Returns:
      float: The average Signal to Noise Ratio for this sequence
    '''
    snrs = []
    for (freq, density), hr in zip(fft, gtHR):
        signal = sum(d**2 for f, d in zip(freq, density) if abs(f-hr/60) <= allowance/60 or abs(f-2*hr/60) <= 2*allowance/60)
        noise = sum(d**2 for f, d in zip(freq, density) if not (abs(f-hr/60) <= allowance/60 or abs(f-2*hr/60) <= 2*allowance/60))
        if noise == 0:
            noise = 0.0000001 # Epsilon
        if signal == 0:
            signal = 0.0000001 # Epsilon
        snrs.append(10 * np.log10(signal/noise))
    if returnAvg:
        return np.nanmean(snrs)
    return snrs

def trimNANs(waves: list) -> list:
    """ Trim all indecies that are NaN in any supplied list
    
    Arguments:
      waves: A list of N equal-length signals that may include NaN values
    
    Returns:
      list: A list of N equal-length signals without NaN values, where their length is <= that of waves
    """
    isnans = [np.isnan(wave) if len(wave.shape)==1 else np.any(np.isnan(wave), axis=-1) for wave in waves]
    notnan = np.invert(np.logical_or.reduce(isnans))
    return [wave[notnan] for wave in waves]

def envelopeCorr(predWave, gtWave, fps, correlate, sign):
    def getEnvelopeSide(wave, flip=False):
        if len(wave.shape) == 2:
            if flip:
                return wave[:,2]
            return wave[:,1]
        else:
            if flip:
                wave *= -1
            if len(predWave.shape) == 2:
                return envelope.peaks(wave, fps, extrapolate=False)
            else:
                return envelope.winmax(wave, fps).numpy()
    if sign == 0:
        envelopes = [np.hstack([getEnvelopeSide(wave, flip=flip) for flip in [False, True]]) for wave in [gtWave, predWave]]
    else:
        envelopes = [getEnvelopeSide(wave, flip=(sign==-1)) for wave in [gtWave, predWave]]
    # Sometimes envelope lengths suffer an off-by-one error
    #envelopes = [env[:min(len(e) for e in envelopes)] for env in envelopes]
    envelopes = trimNANs(envelopes)
    return correlate(*envelopes)

def calcMetrics(args):
    import scipy
    subID, gtHR, predHR, gtWave, predWave, fps, fft, corr = args
    # wrap corr in a lambda that checks for constant inputs:
    correlate = lambda x, y: [0] if (x == x[0]).all() or (y == y[0]).all() else corr(x, y)
    ret = {
            'ME': np.mean(predHR - gtHR),
            'MAE': np.mean(np.abs(gtHR - predHR)),
            'RMSE': np.sqrt(np.mean(np.square(gtHR - predHR))),
            'r_HR': correlate(gtHR, predHR)[0],
            'r_wave': correlate(gtWave, predWave if len(predWave.shape)==1 else predWave[:,0])[0],
            'r_topenv': envelopeCorr(predWave, gtWave, fps, correlate, 1)[0],
            'r_botenv': envelopeCorr(predWave, gtWave, fps, correlate, -1)[0],
            'r_envelope': envelopeCorr(predWave, gtWave, fps, correlate, 0)[0],
            'dhr': dhr(predHR, fps),
            'MXCorr': max(abs(corrs[1]) for corrs in getCorrs(gtWave, predWave if len(predWave.shape)==1 else predWave[:,0], (-int(fps), int(fps))))
        }
    if fft is not None and len(fft) > 0:
        ret['SNR'] = snr(fft, gtHR)
    return subID, ret

def evaluate(predWaves: dict, predHR: dict, gtWaves: dict, gtHR: dict, fps, appendGlobal: bool = False, correlation: str = 'pearsonr', masks: dict = None, fft = None, multiprocessing: bool = True) -> (dict, dict):
    '''Perform an evaluation over the inputs

    All dict arguments are formatted {subjectID: data}

    The returned evaluation metrics are:
      ME: Mean Error
      MAE: Mean Absolute Error
      RMSE: Root Mean Square Error
      r_HR: Pearson r correlation between heart rates
      r_wave: Pearson r correlation between waveforms
      SNR: Signal to Noise Ratio, if fft data provided (optional)

    Arguments:
      predWaves: Predicted waveforms for each subject
      predHR: Preducted heart rate for each subject
      gtWaves: Ground truth waveforms for each subject
      gtHR: Ground truth heart rate for each subject
      appendGlabal: Calculate global metrics by appending individual waves/hr
      correlation: Either 'pearsonr' or 'spearmanr'
      masks: Intervals to mask out for each subject (optional)
      fft: [[freq, density], ...] formatted fft data (optional)
      multiprocessing: Use multiprocessing to speed things up (default=False)

    Returns:
      dict: Evaluation metrics for each subject
      dict: Evaluation metrics averaged over all subjects, weighted by subject length, or from appending results
    '''
    # First, make a (shallow) copy of everything so we don't delete important data
    predWaves = predWaves.copy()
    predHR = predHR.copy()
    gtWaves = gtWaves.copy()
    gtHR = gtHR.copy()
    #predWavesExtra = {}
    #for subID, wave in list(predWaves.items()):
    #    if len(wave.shape) == 2:
    #        predWavesExtra[subID] = wave[1:] #TODO: Utilize
    #        predWaves[subID] = wave[0]
    if masks:
        masks = masks.copy()
    if correlation == 'pearsonr':
        from scipy.stats import pearsonr as corr
    elif correlation == 'spearmanr':
        from scipy.stats import spearmanr as corr
    else:
        print(f'ERROR: Unknown correlation method {correlation}')
        exit(1)
    if masks:
        predWaves, gtWaves, predHR, gtHR = msk.applyMaskDicts([predWaves, gtWaves, predHR, gtHR], masks, method='skip')
        if fft is not None and len(fft) > 0:
            fft, = msk.applyMaskDicts([fft], masks, method='skip')
    # Ensure predicted and gt are same length
    def ensure(pred, gt, name):
        for subID in pred:
            if len(pred[subID]) != len(gt[subID]):
                print(f'WARNING: lengths of {name} for {subID} differ: len(predicted) = {len(pred[subID])}, len(gt) = {len(gt[subID])}')
                minLen = min(len(pred[subID]), len(gt[subID]))
                pred[subID] = pred[subID][:minLen]
                gt[subID] = gt[subID][:minLen]
            # Due to masking, there might be NaNs thrown in. Eliminate those indexes in both.
            pred[subID], gt[subID] = trimNANs([pred[subID], gt[subID]])
        return pred, gt
    predWaves, gtWaves = ensure(predWaves, gtWaves, 'waves')
    predHR, gtHR = ensure(predHR, gtHR, 'HR')
    for subID in list(gtWaves.keys()):
        if any(subID not in dic or len(dic[subID]) < 2 for dic in [predWaves, gtWaves, predHR, gtHR]):
            for dic in [predWaves, gtWaves, predHR, gtHR]:
                if subID in dic:
                    del dic[subID]

    # Obtain by subject
    if multiprocessing:
        import os
        with Pool(int(os.getenv("NSLOTS", default = 0)) or None) as p:
            bySubject = {subID: metrics for subID, metrics in IncrementalBar(f'Calculating evaluation metrics', max=len(predWaves), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(p.imap_unordered(calcMetrics, [(subID, gtHR[subID], predHR[subID], gtWaves[subID], predWaves[subID], fps, None if not fft else fft[subID], corr) for subID in predWaves]))}
    else:
        bySubject = {subID: metrics for subID, metrics in IncrementalBar(f'Calculating evaluation metrics', max=len(predWaves), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(calcMetrics((subID, gtHR[subID], predHR[subID], gtWaves[subID], predWaves[subID], fps, None if not fft else fft[subID], corr)) for subID in predWaves)}
    # Also obtain global values weighted by wave length
    if appendGlobal:
        def append(srcDict, keyOrder):
            res = np.array([])
            for key in keyOrder:
                res = np.append(res, srcDict[key])
            return res
        keyOrder = list(predWaves.keys())
        _, valAvg = calcMetrics((None, append(gtHR, keyOrder), append(predHR, keyOrder), append(gtWaves, keyOrder), append(predWaves, keyOrder), fps, None, corr))
    else:
        totalWaveLen = sum(len(wave) for wave in predWaves.values())
        valAvg = {metric: sum(bySubject[subID][metric] * len(predWaves[subID]) for subID in predWaves) / totalWaveLen for metric in bySubject[list(predWaves.keys())[0]].keys()}
    return bySubject, valAvg

def saveAll(savedir: str, valAvg: dict, bySubject: dict, predWaves: dict, predHR: dict, gtWaves: dict, gtHR: dict, metadata: dict, masks: dict = None):
    if masks:
        # Mess with HR
        predHR, gtHR = msk.applyMaskDicts([predHR, gtHR], masks, method='fill')
    save = Path(savedir)
    save.mkdir(parents=True, exist_ok=True)
    for fname, dic in [['results_bySubject.json', bySubject], ['results_avg.json', valAvg]]:
        with (save / fname).open('w') as f:
            json.dump(dic, f)
    for name, dic in [['waves', predWaves], ['hr', predHR], ['gt_waves', gtWaves], ['gt_hr', gtHR]]:
        dest = save / name
        dest.mkdir(exist_ok=True)
        for subID in dic:
            npio.save(dic[subID], metadata[subID], dest / f'{subID}.npz')
    # Also plot!
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plots = save / 'plots'
    # Mess with waves (ok because already saved)
    for dic in [gtWaves, predWaves]:
        for subID in dic:
            fps = metadata[subID].fps()
            # Cut down to 10 seconds
            dic[subID] = dic[subID][:int(fps*10)]
            # Normalize
            if max(dic[subID]) != min(dic[subID]):
                dic[subID] *= 2/(max(dic[subID]) - min(dic[subID]))
            dic[subID] -= min(dic[subID]) + 1
    for typ, pred, gt, ylabel in [('waves', predWaves, gtWaves, 'Waveform'), ('hr', predHR, gtHR, 'HR (BPM)')]:
        outdir = plots / typ
        outdir.mkdir(parents=True, exist_ok=True)
        for subID in pred:
            fps = metadata[subID].fps()
            plt.figure()
            plt.title(f'{subID} {typ}')
            plt.xlabel('Time (s)')
            plt.ylabel(ylabel)
            for y, label in [(pred[subID], 'pred'), (gt[subID], 'gt')]:
                plt.plot([float(i) / fps for i in range(len(y))], y, label=label)
            plt.legend()
            plt.savefig(outdir / f'{subID}.png')
            plt.close()

if __name__ == '__main__':
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description='Evaluate by comparing to ground truth', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gtWave', help='npz formatted ground truth waveform, or directory of ground truth formatted subID.npz, or test results directory (if only argument), or single wave to evaluate.')
    parser.add_argument('predWave', nargs='?', help='npz formatted predicted waveform, or directory of predictions formatted subID.npz')
    parser.add_argument('gtHR', nargs='?', help='Optionally supply an npz formatted ground truth HR (avoids compututation time)')
    parser.add_argument('predHR', nargs='?', help='Optionally supply an npz formatted predicted HR (avoids compututation time)')
    parser.add_argument('--saveAvg', help='Path to save json-formatted averaged results')
    parser.add_argument('--saveBySubject', help='Path to save json-formatted by-subject results')
    parser.add_argument('--saveAll', help='Path to directory to save all results')
    parser.add_argument('--masks', help='npz formatted masks, or directory of masks formatted subID.npz')
    parser.add_argument('--splits', help='Use a splits.json file')
    parser.add_argument('--split', help='If --splits supplied and gtWave and predWave are directories, sets the split to use', default='test')
    parser.add_argument('--fps', help='Override frames per second of video and waveform as recorded in wave metadata', default=None, type=float)
    parser.add_argument('--hrCalculator', help='Calculator for heart rate', choices=['fft', 'sssp', 'peaks'], default='fft')
    parser.add_argument('--forceRecalculate', help='Recalculate HR even if provided', action='store_true')
    parser.add_argument('--moveCost', help='Parameter when using sssp algorithm', default=10, type=float)
    parser.add_argument('--window', help='Window size for fft HR calculation, in seconds', default=10, type=float)
    parser.add_argument('--binWidth', help='Target width of FFT frequency bins; nfft is set based on this value', default=0.001, type=float)
    parser.add_argument('--dequantize', help='Dequantize FFT data using dequantize subroutine', action='store_true')
    parser.add_argument('--skipFFT', action='store_true', help='Skips calculation of FFT (and SNR)')
    parser.add_argument('--lowHz', help='Low bound of frequency for HR', default=0.66666, type=float)
    parser.add_argument('--highHz', help='High bound of frequency for HR', default=3, type=float)
    parser.add_argument('--deltaLimit', help='Limit change in HR; units of bpm/second (occurs before smooth if specified)', type=float)
    parser.add_argument('--smooth', help='Smooth using a sliding window; parameter is window width in seconds (occurs after deltaLimit if specified)', type=float)
    parser.add_argument('--shift', help='Maximize r_wave by shifting predWave up to the specified amount in seconds.', type=float, default=0)
    parser.add_argument('--appendGlobal', action='store_true', help='Calculate global metrics by appending individual waves/hr')
    parser.add_argument('--correlation', help='Method for calculating the correlation coefficient', choices=['pearsonr', 'spearmanr'], default='pearsonr')
    parser.add_argument('--predHRAvg', action='store_true', help='Rather than calculate a predicted HR, use the average HR of the entire dataset.')
    parser.add_argument('--singleprocess', action='store_true', help='Force singleprocessing (helpful on some machines)')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    # Load masks
    masks = {}
    if args.masks:
        if Path(args.masks).is_dir():
            masks = {fname.stem: npio.load(fname) for fname in Path(args.masks).iterdir()}
        else: # Is actually a file
            masks = {'wave': npio.load(args.masks)}

    # Load the waves
    if not args.predWave:
        if Path(args.gtWave).is_dir() and Path(args.gtWave + '/waves').is_dir(): # We're given the root test directory as args.gtWave
            args.predWave = args.gtWave + '/waves'
            args.gtHR = args.gtWave + '/gt_hr'
            args.predHR = args.gtWave + '/hr'
            args.gtWave = args.gtWave + '/gt_waves'
        else: # We're given a single wave to analyze
            pass
    if args.forceRecalculate:
        args.gtHR = None
        args.predHR = None
    if args.splits:
        with open(args.splits) as f:
            subIDs = json.load(f)[args.split]
    waves = {}
    metas = {}
    fps = args.fps
    for name in ['gtWave', 'predWave']:
        if not getattr(args, name):
            continue
        waves[name] = {}
        path = Path(getattr(args, name))
        fnames = path.iterdir() if path.is_dir() else [path]
        for fname in fnames:
            subID = fname.stem if path.is_dir() else 'wave'
            if args.splits and subID not in subIDs:
                continue
            wave, meta = npio.loadMeta(fname)
            waves[name][subID] = wave
            metas[subID] = meta
            if meta:
                thisFPS = meta.fps()
                if not fps:
                    fps = thisFPS
                elif abs(fps - thisFPS) > 0.1:
                    print(f'ERROR: fps {thisFPS} and {fps} differ!')
                    #exit(1)
    
    # Ensure waves are same subset
    common = set.intersection(*[set(w.keys()) for w in waves.values()])
    waves = {name: {subID: wave for subID, wave in waves[name].items() if subID in common} for name in waves}

    # Shift waves
    if args.shift:
        for subID in waves['predWave']:
            if subID not in waves['gtWave']:
                continue
            bestShift = getShift(waves['gtWave'][subID], waves['predWave'][subID], args.shift*fps)
            if bestShift > 0:
                waves['gtWave'][subID] = waves['gtWave'][subID][bestShift:]
            else:
                waves['predWave'][subID] = waves['predWave'][subID][-bestShift:]

    # Calculate heart rates
    HRs = {}
    FFT = {}
    for name in waves:
        HRs[name] = {}
        if getattr(args, name.replace('Wave', 'HR')):
            path = Path(getattr(args, name.replace('Wave', 'HR')))
            fnames = path.iterdir() if path.is_dir() else [path]
            for fname in fnames:
                subID = fname.stem if path.is_dir() else 'wave'
                if args.splits and subID not in subIDs:
                    continue
                HRs[name][subID] = npio.load(fname)
        else:
            hrAndFFT = hr.calcHR(waves[name], fps, hrCalculator=args.hrCalculator, lowHz=args.lowHz, highHz=args.highHz, binWidth=args.binWidth, window=args.window, dequantize=args.dequantize, moveCost=args.moveCost, deltaLimit=args.deltaLimit, smooth=args.smooth, skipFFT=args.skipFFT)
            for subID, (HR, fft) in hrAndFFT.items():
                HRs[name][subID] = HR
                if len(fft) > 0 and hasattr(fft[0], '__len__'):
                    if name not in FFT:
                        FFT[name] = {}
                    FFT[name][subID] = fft

    if args.predHRAvg:
        # Replace predicted HRs with the average of the GT HRs.
        avgHR = np.mean([h for hr in HRs['gtWave'].values() for h in hr])
        print(f'Average HR: {avgHR}')
        for subID, gt in HRs['gtWave'].items():
            HRs['predWave'][subID] = [avgHR] * len(gt)
    
    # Ensure hrs are same subset
    common = set.intersection(*[set(h.keys()) for h in HRs.values()])
    HRs = {name: {subID: HR for subID, HR in HRs[name].items() if subID in common} for name in HRs}

    predWave = waves['gtWave'] if 'predWave' not in waves else waves['predWave']
    predHR = HRs['gtWave'] if 'predWave' not in HRs else HRs['predWave']
    fft = None if not FFT else FFT['predWave'] if 'predWave' in FFT else FFT['gtWave']
    bySubject, valAvg = evaluate(predWave, predHR, waves['gtWave'], HRs['gtWave'], fps, args.appendGlobal, masks=masks, fft=fft, multiprocessing=not args.singleprocess)
    for fname, dic in [[args.saveAvg, valAvg], [args.saveBySubject, bySubject]]:
        if fname:
            with open(fname, 'w') as f:
                json.dump(dic, f)
    if args.saveAll:
        saveAll(args.saveAll, valAvg, bySubject, predWave, predHR, waves['gtWave'], HRs['gtWave'], metas, masks)
    metrics = list(valAvg.keys())
    from rPPG.utils import tables
    # Format tabular data as {header: values}
    data = {'SubID': list(sorted([subID for subID in bySubject])) + ['avg']}
    for metric in valAvg.keys():
        data[metric] = [bySubject[subID][metric] for subID in data['SubID'][:-1]] + [valAvg[metric]]
    tables.printTable(data)
