#!/usr/bin/env python3

import numpy as np
from scipy import signal

def smooth(HR: np.ndarray, fps: float, width: float = 5) -> np.ndarray:
    """Smooth the sample using a sliding window

    Args:
      HR: The unsmoothed heart rate as an array of samples
      fps: The framerate (sampling rate) of HR
      width: The window width, in seconds (default: 5 seconds)

    Returns:
      np.ndarray: The smoothed heart rate
    """
    maf_width=int(width * fps)
    return np.convolve(np.pad(HR, maf_width//2, 'edge'), np.ones((maf_width))/maf_width, mode='valid')

def rateLimit(HR: np.ndarray, fps: float, maxDelta: float = 1, ignore: float = 5) -> np.ndarray:
    """Limit the rate of change of the heartrate

    Args:
      HR: The unlimited heart rate as an array of samples
      fps: The framerate (sampling rate) of HR
      maxDelta: The limit to the change in heartrate in bpm/second
      ignore: The amount of data to let pass before enforcing maxDelta, in seconds

    Returns:
      np.ndarray: The limited heart rate
    """
    if len(HR) == 0:
        return []
    maxD = maxDelta/fps
    ret = [HR[0]]
    for i, hr in enumerate(HR[1:]):
        if i < fps*ignore or abs(ret[-1]-hr) < maxD:
            ret.append(hr)
        else:
            if hr < ret[-1]:
                ret.append(ret[-1]-maxD)
            else:
                ret.append(ret[-1]+maxD)
    return ret

def dequantizePeaks(x: np.ndarray, y: np.ndarray):
    peaks = signal.find_peaks(y)[0]
    if len(peaks) == 0:
        return [], []
    valleys = signal.find_peaks([-1 * Y for Y in y])[0]
    retX = []
    for peak in peaks:
        # Average over adjacent valleys
        val1 = 0
        if any(v < peak for v in valleys):
            val1 = max(v for v in valleys if v < peak)
        val2 = len(x)
        if any(v > peak for v in valleys):
            val2 = min(v for v in valleys if v > peak)+1
        retX.append(sum(x[i] * y[i] for i in range(val1, val2)) / sum(y[i] for i in range(val1, val2)))
    return zip(retX, [y[peak] for peak in peaks])

def calcHRFFT(wave: np.ndarray, fps: float, window: float, lowHz: float = 0.66666, highHz: float = 3, fft: list = None, dequantize: bool = False, binWidth: float = 0.001, skipFFT: bool = False) -> (np.ndarray, np.ndarray):
    """Calculate the heartrate using FFT with a sliding window

    Args:
      wave: The waveform as an array of samples
      fps: The framerate (sampling rate) of wave
      window: The window size in seconds used for calculating the FFT
      lowHz: The lower limit for reported heartate in hz
      highHz: The upper limit for reported heartrate in hz
      fft: Use cached fft data rather than recalculate
      dequantize: Use dequantizePeaks method
      binWidth: Target width of frequency bins; nfft is set based on this value. If zero, then nfft is left as default
      skipFFT: If true, return empty array in place of fft (useful on low-memory machines)

    Returns:
      np.ndarray: The heart rate
      np.ndarray: FFT data at each frame
    """
    HR = []
    retFFT = []
    windowLen = min(int(fps * window), len(wave)-1)
    window = signal.get_window('hamming', windowLen)
    for fftIndex, i in enumerate(range(len(wave) - windowLen + 1)):
        if fft:
            freq, density = fft[fftIndex]
        else:
            nfft = None if not binWidth else int(fps/binWidth)
            freq, density = signal.periodogram(np.array(wave[i:i+windowLen]), window=window, fs=fps, nfft=nfft)
            idcs = np.where((freq >= lowHz) & (freq <= highHz))[0]
            freq = freq[idcs]
            density = density[idcs]
        if not skipFFT:
            retFFT.append(np.array([freq, density]))
        if dequantize:
            try:
                freq, density = zip(*dequantizePeaks(freq, density))
            except ValueError:
                if len(HR) > 0:
                    HR.append(HR[-1])
                else:
                    HR.append(70)
                continue
        highest = np.argmax(density)
        HR.append(freq[highest] * 60)

    return HR, np.array(retFFT)

def calcHRSSSP(fft: np.ndarray, fps: float, moveCost: float) -> np.ndarray:
    ''' Use the single source shortest path algorithm to calculate HR

    Args:
      fft: The FFT data at each frame as returned by calcHRFFT
      fps: The framerate of the FFT data in frames per second
      moveCost: The cost to change HR. Units are cumulative strength * minute

    Returns:
      np.ndarray: The heart rate
    '''
    # Convert moveCost from cumulative strength * minute to strength * frame
    moveCost *= 60 * fps
    optimal = []
    for x, y in fft:
        optimalNext = []
        try:
            x, y = zip(*dequantizePeaks(x, y))
        except ValueError:
            pass
        for freq, strength in zip(x, y):
            child = {'freq': freq}
            if not optimal:
                child['score'] = -1 * strength
                child['parent'] = None
            else:
                child['parent'], child['score'] = min(zip(optimal, [(moveCost*(freq-opt['freq'])**2)+opt['score']-strength for opt in optimal]), key=lambda zipped: zipped[1])
            optimalNext.append(child)
        optimal = optimalNext
    shortest = min(optimal, key=lambda opt: opt['score'])
    HR = []
    while shortest is not None:
        HR.append(shortest['freq'] * 60)
        shortest = shortest['parent']
    return list(reversed(HR))

def _peaksGenerator(waveGenerator, fps, lowHz = 0.66666, highHz = 3):
    """Helper function for calcHRPeaks"""
    # Grab from waveGenerator until we identify a peak, then yield its index.
    frameMin = int(fps/highHz)
    frameMax = int(fps/lowHz)
    # We enforce highHz and lowHz by reading frameMax frames from the last yielded one into memory, and selecting one at least frameMin in.
    activeFrames = []
    index = 0
    def findPeak(frames):
        # Find the one between frameMin and frameMax
        prominence = (np.percentile(frames, 75) - np.percentile(frames, 25)) / 2
        peaks, _ = signal.find_peaks(frames, distance=frameMin, prominence=prominence)
        while len(peaks) > 0 and peaks[0] < frameMin:
            peaks = peaks[1:]
        return len(frames)-1 if len(peaks) == 0 else peaks[0]

    for w in waveGenerator:
        activeFrames.append(w)
        if len(activeFrames) > frameMax:
            maxI = findPeak(activeFrames)
            yield index + maxI
            activeFrames = activeFrames[maxI:]
            index += maxI
    while len(activeFrames) > frameMin:
        maxI = findPeak(activeFrames)
        if maxI == len(activeFrames)-1: # Don't trust it
            return
        yield index + maxI
        activeFrames = activeFrames[maxI:]
        index += maxI

def calcHRPeaks(wave: np.ndarray, fps: float, lowHz: float = 0.66666, highHz: float = 3, binWidth: float = 0, backend: str = 'scipy') -> (np.ndarray, np.ndarray):
    """Calculate the heartrate using peak detection

    Args:
      wave: The waveform as an array of samples
      fps: The framerate (sampling rate) of wave
      lowHz: The lower limit for reported heartate in hz
      highHz: The upper limit for reported heartrate in hz
      binWidth: Target width of frequency bins; cubic spline interpolation is used to achieve this. If zero, then no interpolation is applied. Actually, never interpolate.
      backend: Either scipy or pyampd

    Returns:
      np.ndarray: The heart rate
      np.ndarray: Indices of the detected peaks
    """
    '''if binWidth != 0:
        # Adjust wave and fps
        from scipy.interpolate import CubicSpline
        x = [i / fps for i in range(len(wave))]
        spline = CubicSpline(x, wave)
        fps = 1 / binWidth
        wave = spline(np.arange(0, x[-1], 1/fps))'''
    if backend == 'pyampd':
        from pyampd.ampd import find_peaks
        peaks = find_peaks(wave)
    elif backend == 'scipy':
        peaks = list(_peaksGenerator(wave, fps, lowHz, highHz))
    else:
        raise ValueError(f'Unknown peak detection backend: {backend}')
    # Don't trust peaks at the very start nor end
    if peaks[0] == 0 and len(peaks) > 2:
        peaks = peaks[1:]
    if peaks[-1] == len(wave)-1 and len(peaks) > 2:
        peaks = peaks[:-1]
    HR = sum([[fps / (nxt-cur) * 60] * (nxt-cur) for cur, nxt in zip(peaks, peaks[1:])], [])
    if len(wave) > len(HR):
        if len(HR) == 0:
            HR = [0]
        # Extend HR to be the same length as wave
        HR = HR + ([HR[-1]] * (len(wave) - len(HR)))
    return np.array(HR), peaks

def calcHRCWT(wave: np.ndarray, fps: float, lowHz: float = 0.66666, highHz: float = 3, binWidth: float = 0.001) -> (np.ndarray, np.ndarray):
    """Calculate the heartrate using a continuous wavelet transform

    Args:
      wave: The waveform as an array of samples
      fps: The framerate (sampling rate) of wave
      lowHz: The lower limit for reported heartate in hz
      highHz: The upper limit for reported heartrate in hz
      binWidth: Target width of frequency bins; cubic spline interpolation is used to achieve this. If zero, then no interpolation is applied.

    Returns:
      np.ndarray: The heart rate
      np.ndarray: The CWT data at each frame
    """
    import ssqueezepy
    from scipy.interpolate import interp1d
    #Tx, Wx, ssq_freqs, scales = ssqueezepy.ssq_cwt(wave, fs=fps, maprange=(lowHz, highHz), scales='linear')
    Tx, Wx, ssq_freqs, scales = ssqueezepy.ssq_cwt(wave, fs=fps, nv=128)
    penalty=10000 # Maybe analagous to sssp?
    bw=25 # Maybe only applies when n_ridges > 1?
    inbounds = np.where((ssq_freqs >= lowHz)*(ssq_freqs <= highHz)) # * is logical and
    x = ssq_freqs[inbounds]
    xs = scales[inbounds]
    Tf = Tx[inbounds]
    ridge_idxs = ssqueezepy.extract_ridges(Tf, xs, penalty, n_ridges=1, bw=bw)
    HR = np.array([x[idxs[0]]*60 for idxs in ridge_idxs]) # ridge_idxs is [frame x n_ridges] 2d array
    Tf = interp1d(x, np.abs(Tf), axis=0)(np.linspace(x[0], x[-1], len(x)))
    return HR, np.abs(Tf)

def __calcHR(args):
    subID, wave, fps, hrCalculator, lowHz, highHz, window, dequantize, binWidth, skipFFT, moveCost, deltaLimit, sm = args
    fftOrPeaks = None
    if hrCalculator.startswith('peaks'):
        if hrCalculator == 'peaks':
            HR, fftOrPeaks = calcHRPeaks(wave, fps, lowHz, highHz)
        else:
            HR, fftOrPeaks = calcHRPeaks(wave, fps, lowHz, highHz, backend=hrCalculator.split('-')[-1])
    elif hrCalculator == 'cwt':
        HR, fftOrPeaks = calcHRCWT(wave, fps, lowHz, highHz)
    else:
        HR, fftOrPeaks = calcHRFFT(wave, fps, window, lowHz, highHz, dequantize=dequantize, binWidth=binWidth, skipFFT=False if hrCalculator == 'sssp' else skipFFT)
        if hrCalculator == 'sssp':
            HR = calcHRSSSP(fftOrPeaks, fps, moveCost)
    if deltaLimit:
        HR = rateLimit(HR, fps, maxDelta=deltaLimit)
    if sm:
        HR = smooth(HR, fps, width=sm)
    return subID, np.array(HR), fftOrPeaks

def calcHR(waves: dict, fps: float, hrCalculator: str = 'fft', lowHz: float = 0.66666, highHz: float = 3, window: float = 10, dequantize: bool = False, binWidth: float = 0.001, skipFFT: bool = False, moveCost: float = 10, deltaLimit: float = 0, smooth: float = 0) -> dict:
    '''Perform HR Calculation using multiprocessing

    Args:
      waves: A dict of {subID: waveform}
      fps: The framerate (sampling rate) of all waves
      hrCalculator: Any of fft, sssp, peaks-scipy, or peaks-pyampd
      lowHz: Low frequency in bandpass filtering techniques
      highHz: High frequency in bandpass filtering techniques
      window: Window size, in seconds, for fft
      dequantize: Apply the dequantizing subroutine for fft and sssp
      binWidth: Target width of frequency bins; nfft is set based on this value
      skipFFT: If true, return None in place of fft (useful on low-memory machines)
      moveCost: Movement cost for sssp
      deltaLimit: If not 0, apply rateLimit routine at this rate
      smooth: If not 0, apply smooth routine with this window

    Returns:
      dict: A dict of {subID: (hr, fft)}
    '''
    allArgs = [(subID, wave, fps, hrCalculator, lowHz, highHz, window, dequantize, binWidth, skipFFT, moveCost, deltaLimit, smooth) for subID, wave in waves.items()]
    from progress.bar import IncrementalBar
    from multiprocessing import Pool
    import os
    with Pool(int(os.getenv("NSLOTS", default = 0)) or None) as p:
        return {subID: (hr, fft) for subID, hr, fft in IncrementalBar('Calculating HR', max=len(waves), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(p.imap_unordered(__calcHR, allArgs))}
