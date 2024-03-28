#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import numpy as np
from rPPG.utils import hr
from rPPG.utils import evaluate

def applyMaskFill(arry: list, mask: list, offset: int = 0, value = np.nan) -> list:
    '''Apply a mask to an array using a fill value. This method keeps the same array length.

    Arguments:
      arry: The array to be masked
      mask: An array of intervals denoting indices to be masked
      offset: An offset by which to shift mask indices (optional)
      value: Fill value

    Returns:
      list: The masked array
    '''
    ret = np.array(arry)
    if len(ret) <= 1: # Messes up len(ret)-1 -> start-1
        return ret
    for start, end in mask:
        start, end = [min(max(0, x+offset), len(ret)-1) for x in [start, end]]
        ret[start:end] = value
    return ret

def applyMaskLinear(arry: list, mask: list, offset: int = 0) -> list:
    '''Apply a mask to an array using linear interpolation. This method keeps the same array length.

    Arguments:
      arry: The array to be masked
      mask: An array of intervals denoting indices to be masked
      offset: An offset by which to shift mask indices (optional)

    Returns:
      list: The masked array
    '''
    ret = np.array(arry)
    if len(ret) <= 1: # Messes up len(ret)-1 -> start-1
        return ret
    for start, end in mask:
        start, end = [min(max(0, x+offset), len(ret)-1) for x in [start, end]]
        y1 = ret[end] if start == 0 else ret[start]
        y2 = ret[start] if end == len(ret)-1 else ret[end]
        ret[start:end] = np.linspace(y1, y2, end-start)
    return ret

def applyMaskSkip(arry: list, mask: list, offset: int = 0) -> list:
    '''Apply a mask to an array by skipping masked values. This method returns an array that may be shorter than the input array.

    Arguments:
      arry: The array to be masked
      mask: An array of intervals denoting indices to be masked
      offset: An offset by which to shift mask indices (optional)

    Returns:
      list: The masked array
    '''
    return np.array([val for i, val in enumerate(arry) if not any(i >= start+offset and i < end+offset for start, end in mask)])

def applyMaskMulti(arrys: list, mask: list, method='linear') -> (list):
    '''Apply a mask to multiple arrays, inferring the offset based on the difference between array lengths. The longest of the arrays will have offset=0.

    Arguments:
      arrys: Input arrays, not necessarily the same length
      mask: An array of intervals denoting indices to be masked
      method: The method to use; 'linear', 'skip', or 'fill' (default: 'linear')

    Returns:
      list: Masked versions of arrys
    '''
    if method == 'linear':
        maskMethod = applyMaskLinear
    elif method == 'skip':
        maskMethod = applyMaskSkip
    elif method == 'fill':
        maskMethod = applyMaskFill
    else:
        raise ValueError(f'Unknown mask method: {method}')
    longLen = max(len(arry) for arry in arrys)
    return [maskMethod(arry, mask, offset=(len(arry)-longLen)//2) for arry in arrys]

def applyMaskDicts(dicts: list, masks: dict, method='linear') -> list:
    '''Convenience function to access the functionality of applyMaskMulti'''
    retDicts = [{} for dic in dicts]
    for subID, mask in masks.items():
        if not all(subID in dic for dic in dicts):
            continue
        for i, arry in enumerate(applyMaskMulti([dic[subID] for dic in dicts], mask, method)):
            retDicts[i][subID] = arry
    return retDicts

def calcMask(wave: list, fps: float, HR: list = None, fft: list = None, window: float = 10, maxDelta: float = 7, snrs: list = None, snrCutoff: float = -.5, criterion='dhr'):
    ''' Calculate a mask for a wave

    Arguments:
      wave: Waveform to calculate a mask over
      fps: Sample rate of the wave
      HR: Cached pre-calculated HR
      fft: Cached pre-calucaled FFT
      window: Amout of time in seconds to mask out surrounding the noisy section
      maxDelta: HR delta criteria for segment to be classified as noisy
    '''
    if HR is None or (fft is None and criterion == 'snr'):
        HR, fft = hr.calcHRFFT(wave, fps, window, fft=fft)
    remove = []
    for i, (bpm, fftWindow) in enumerate(zip(HR, fft)):
        if criterion == 'dhr':
            fails = any(abs(bpm-other) > maxDelta for other in HR[max(0, int(i-fps)):min(len(HR), int(i+fps))])
        elif criterion == 'snr':
            if snrs is None:
                snrs = evaluate.snr(fft, HR)
            fails = snrs[i] < snrCutoff
        else:
            raise NotImplementedError(f'Mask criterion {criterion} not implemented.')
        if fails:
            if remove and remove[-1][1] >= i-window*fps/2:
                remove[-1][1] = min(int(i+window*fps/2), len(HR))
            else:
                remove.append([max(int(i-window*fps/2), 0), min(int(i+window*fps/2), len(HR))])

    buffer = int((len(wave)-len(HR))/2)
    remove = [[interval[0]+buffer, interval[1]+buffer] for interval in remove]
    waveClean = [-1 if any(i >= interval[0] and i < interval[1] for interval in remove) else val for i, val in enumerate(wave)]
    return remove, waveClean

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Apply masks to clean signal sections that are suspected to be in error', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='npz formatted signal to clean')
    parser.add_argument('--outIntervals', help='Output error sections as an npz file indicating intervals')
    parser.add_argument('--outLabel', help='Output a label with error sections set to 0')
    parser.add_argument('--outPlots', help='Save output diagnostic plots to given directory')
    parser.add_argument('--loadFFT', help='Load cached fft frequency/density data')
    parser.add_argument('--loadHR', help='Load cached HR data')
    parser.add_argument('--window', help='Window size for fft HR calculation, in seconds', default=10, type=float)
    parser.add_argument('--maxDelta', help='Maximum change in HR, units of bpm/second, used to identify error sections of waveform', default=7, type=float)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from rPPG.utils import npio
    from pathlib import Path
    inputPath = Path(args.input)
    wave, metadata = npio.loadMeta(inputPath)
    fps = metadata.fps()
    fft = None
    HR = None
    if args.loadFFT:
        fft = npio.load(args.loadFFT)
    if args.loadHR:
        HR = npio.load(args.loadHR)
    
    remove, waveClean = calcMask(wave, fps, HR, fft, args.window, args.maxDelta)

    # Output stuff

    if args.outIntervals:
        print(f'Outputting to {args.outIntervals}')
        npio.save(remove, metadata, args.outIntervals)
    if args.outLabel:
        npio.save(waveClean, metadata, args.outLabel)
    if args.outPlots:
        import matplotlib.pyplot as plt
        outPlots = Path(args.outPlots)
        outPlots.mkdir(parents=True, exist_ok=True)
        intervals = outPlots / (inputPath.name + '-intervals/')
        intervals.mkdir(parents=True, exist_ok=True)
        for interval in remove:
            # We plot the interval's HR plus waveform
            print(f'Plotting interval {interval}')
            hrInterval = [i-buffer for i in interval]
            plt.figure()
            plt.title(f'Heart Rate for Error Interval {interval[0]/fps:.4f} to {interval[1]/fps:.4f} seconds')
            plt.xlabel('Time (s)')
            plt.ylabel('Heart Rate (bpm)')
            x = [i/fps for i in range(*interval)]
            y = HR[hrInterval[0]:hrInterval[1]]
            #print(f'Dealing with hrInterval: {hrInterval}, and HR has len: {len(HR)}')
            plt.plot(x, y)
            plt.savefig(intervals / f'hr-{interval[0]/fps:.4f}-{interval[1]/fps:.4f}.png')
            plt.figure()
            plt.title(f'Waveform for Error Interval {interval[0]/fps:.4f} to {interval[1]/fps:.4f} seconds')
            plt.xlabel('Time (s)')
            plt.ylabel('Waveform')
            y = wave[interval[0]:interval[1]]
            #print(f'Dealing with interval: {interval}, and wave has len: {len(wave)}')
            plt.plot(x, y)
            plt.savefig(intervals / f'wave-{interval[0]/fps:.4f}-{interval[1]/fps:.4f}.png')
            plt.close('all')
        # And now, plot the overall situation:
        plt.figure()
        plt.title('Original and Corrected Heart Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate (bpm)')
        x = [i/fps for i in range(buffer, len(HR)+buffer)]
        plt.plot(x, HR, label='Original')
        corrected = [np.nan if any(i+buffer >= interval[0] and i+buffer < interval[1] for interval in remove) else val for i, val in enumerate(HR)]
        plt.plot(x, corrected, label='Corrected')
        plt.legend()
        plt.savefig(outPlots / (inputPath.name + '-hr-full.png'))
