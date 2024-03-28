#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

def calcRWindowed(gt: list, baselines: list, shiftSearch: tuple, window: int, stride: int) -> list:
    """Calculates the Pearson R correlation for shifting gt at stride-spaced windows

    Args:
      gt: The gt to shift
      baselines: The list of baseline waveforms to consider
      shiftSearch: 2-length tuple of bounds to consider for shifting
      window: Number of frames over which to calculate R
      stride: Stride length to move window

    Returns:
      list: One element per window of dicts of form: {shift (int): r (float)}
    """
    from rPPG.utils import evaluate
    return [
            {shift[0]: sum(r) for shift, r in
                [zip(*c) for c in zip(*[evaluate.getCorrs(baseline[i:i+window], gt[i:i+window], shiftSearch) for baseline in baselines])]}
            for i in range(0, min(len(l) for l in [gt] + baselines)-window, stride)]

def debugPlots(windows, shifts, out, prefix='', shiftsGlobal=[]):
    from pathlib import Path
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    import matplotlib.pyplot as plt
    from pathlib import Path
    if len(shiftsGlobal) > 0:
        plt.figure()
        plt.title(f'{prefix} Shift Search')
        plt.xlabel('Shift (frames)')
        plt.ylabel('r')
        X = []
        Y = []
        for i, sb in enumerate(shiftsGlobal):
            x = [s for s in sb.keys()]
            y = [r for r in sb.values()]
            X.append(x)
            Y.append(y)
            plt.plot(x, y, label=f'Baseline {i}')
        if len(shiftsGlobal) > 1:
            plt.plot(X[0], [sum(y)/len(y) for y in zip(*Y)], label='Average')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f'{prefix}shiftsGlobal.png')
    plt.figure()
    plt.title(f'{prefix} Windows')
    plt.xlabel('Window number')
    plt.ylabel('Shift')
    C = [[window[shift] for window in windows] for shift in windows[0]]
    X = list(range(len(windows)))
    Y = [shift for shift in windows[0]]
    plt.pcolormesh(X, Y, C, shading='gouraud')
    plt.plot(shifts, color='red')
    plt.tight_layout()
    plt.savefig(out / f'{prefix}windows.png')
    '''plt.figure()
    plt.title(f'{prefix} Windows')
    plt.xlabel('Window number')
    plt.ylabel('Shift')
    plt.plot(shifts, color='red')
    plt.tight_layout()
    plt.savefig(out / f'{prefix}shifts.png')'''

def shiftSignal(gt: list, baselines: list, maxShift: int, fps: float, temporal: bool = False, plotDir: str = None, plotPrefix: str = '') -> list:
    """Shifts a signal up to maxShift based on baselines

    Args:
      gt: The gt to shift
      baselines: The baseline waveforms to consider when shifting
      maxShift: The maximum number of frames to shift the gt
      temporal: If true, use windowed technique to compress/expand temporally
      plotDir: Directory to save debugging plots (default = None)
      plotPrefix: String prefixed to plot file names

    Returns:
      list: The shifted gt
    """
    from rPPG.utils import evaluate
    window = min(len(l) for l in [gt] + baselines)-1
    stride = window
    shiftsGlobalByBaseline=[]
    if maxShift == 0: # Set maxShift based on global shift data
        shiftsGlobalByBaseline = [calcRWindowed(gt, [b], (-int(fps*1.5), int(fps*1.5)), window, stride)[0] for b in baselines]
        shifts = {s[0]: sum(r) for s, r in [zip(*sr) for sr in zip(*[s.items() for s in shiftsGlobalByBaseline])]}
        def goToExtrema(start, comparator):
            for move in [-1, 1]:
                if start+move in shifts and comparator(shifts[start+move], shifts[start]):
                    return goToExtrema(start+move, comparator)
            return start
        import operator
        peak = goToExtrema(0, operator.gt)
        maxShift = [goToExtrema(s, operator.lt) for s in [peak-1, peak+1]]
    else:
        maxShift = (-maxShift, maxShift)
    #print(f'Using shift interval: {maxShift}')
    if temporal:
        from scipy import signal
        window = int(10*fps)
        stride = window
    windows = calcRWindowed(gt, baselines, maxShift, window, stride)
    gtShifted = []
    cumShift = 0
    # Apply sliding average, weighted by pearson r^2
    slidingAverage = min(9, len(windows))
    shifts = [max(shifts.items(), key=lambda sr: sr[1]) for shifts in windows]
    shifts = [sum(s[0]*(s[1]**2) for s in sa) / sum(s[1]**2 for s in sa) for sa in [shifts[i:i+slidingAverage] for i in range(len(shifts)-slidingAverage+1)]]
    #print(f'Shifts: {shifts}')
    #shifts = [sum(s[0] for s in sa) / len(sa) for sa in [shifts[i:i+slidingAverage] for i in range(len(shifts)-slidingAverage+1)]]
    shifts = [shifts[0]] * int((slidingAverage-1)/2) + shifts + [shifts[-1]] * int((slidingAverage-1)/2)
    shifts = [int(round(s)) for s in shifts]
    #from rPPG.utils import hr
    #shifts = [round(s).astype(int) for s in hr.smooth(shiftsUnsmoothed, 1, slidingAverage)]
    # For window 0: shift entire signal
    # For windows 1-end: compress/expand window
    for w, shift in enumerate(shifts):
        w *= stride
        shift -= cumShift
        #print(f'Shifting by {shift}')
        if w == 0: # base case
            if shift > 0:
                gtShifted = [0] * int(round(shift)) + list(gt)[:window]
            else:
                gtShifted = list(gt[-shift:window])
        else: # Compress/expand
            gtShifted += list(signal.resample(gt[w:w+window], window+shift))
        cumShift += shift
    if plotDir:
        debugPlots(windows, shifts, plotDir, plotPrefix, shiftsGlobalByBaseline)
    return gtShifted

def __calcShifted(args):
    name, gt, signals, baselines, maxShift, temporal, debug = args
    from rPPG.baselines import calcBaselines
    blines = {}
    blines['CHROM'], blines['POS'], meta = calcBaselines.loadAndCalculate(signals)
    gtShifted = shiftSignal(gt, [blines[b] for b in baselines], int(maxShift*meta.fps()), meta.fps(), temporal, debug, name)
    return name, gtShifted, meta

def calcShifted(gt: dict, signals: dict, baselines: list = ['CHROM', 'POS'], maxShift: float = 0, temporal: bool = False, debug: str = '') -> dict:
    '''Perform the shift calculation

    Arguments:
      gt: A dict of {subID: waveform}
      signals: A dict of {subID: path}
      baselines: List of baselines to use
      maxShift: Maximum amonut to shift gt, in seconds. Default: Automatically determine
      temporal: Apply temporal distortion
      debug: Plot debugging output

    Returns:
      dict: A dict of {subID: (shifted, metadata)}
    '''
    allArgs = [(subID, gt[subID], signals[subID], baselines, maxShift, temporal, debug) for subID in gt]
    from progress.bar import IncrementalBar
    from multiprocessing import Pool
    import os
    with Pool(int(os.getenv("NSLOTS", default = 0)) or None) as p:
        return {subID: (shifted, meta) for subID, shifted, meta in IncrementalBar('Shifting gt', max=len(gt), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(p.imap_unordered(__calcShifted, allArgs))}

if __name__ == '__main__':
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description='Adjust ground truth signals using CHROM or POS', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gt', help='Input ground truth signal or dir')
    parser.add_argument('signals', help='Path to spatially averaged video signals')
    parser.add_argument('output', help='Adjusted signal')
    parser.add_argument('--baseline', nargs='+', choices=['CHROM', 'POS'], default=['CHROM'], help='Baseline technique to use for adjusting gts')
    parser.add_argument('--maxShift', type=float, default=0, help='Maximum amount to shift gt, in seconds. Default: Automatically determine')
    parser.add_argument('--temporal', action='store_true', help='Apply temporal distortion')
    parser.add_argument('--debug', help='Plot debugging output')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from rPPG.utils import npio
    from pathlib import Path
    
    src = Path(args.gt)
    signals = Path(args.signals)
    output = Path(args.output)
    if src.is_dir():
        src = list(src.iterdir())
        signals = {s.name: signals / s.name for s in src if (signals/s.name).exists()}
        src = [s for s in src if s.name in signals]
        output.mkdir(exist_ok=True, parents=True)
    else:
        signals = {src.name: signals}
        src = [src]
    
    gt = {s.name: npio.load(s) for s in src}

    shifted = calcShifted(gt, signals, baselines = args.baseline, maxShift = args.maxShift, temporal = args.temporal, debug = args.debug)

    for name, (gtShifted, meta) in shifted.items():
        out = output / name if output.is_dir() else output
        npio.save(gtShifted, meta, out)
