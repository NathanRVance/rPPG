#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Convert the waveform to rate estimates', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('waveform', nargs='+', help='npz formatted waveform. If multiple are given, then output, saveFFT, and plotPeaks are prepended by index. If a directory is given, then a directory is expected for output.')
    parser.add_argument('output', help='Location to save npz formatted calculated HR')
    parser.add_argument('--hrCalculator', help='Calculator for heart rate', choices=['fft', 'sssp', 'peaks-scipy', 'peaks-pyampd', 'cwt'], default='fft')
    parser.add_argument('--plotHR', help='Location to save plotted HR')
    parser.add_argument('--plotHR-ylims', help='If run with --plotHR, set the LOW and HIGH bounds, in bpm (default: automatic y range)', nargs=2, type=float, metavar=('LOW', 'HIGH'))
    parser.add_argument('--plotHR-title', help='Title on plotted HR, default="HR for {args.hrCalculator}"', default=None)
    parser.add_argument('--saveFFT', help='Save fft frequency/density data to PATH', metavar='PATH')
    #parser.add_argument('--loadFFT', help='Load cached fft frequency/density data')
    parser.add_argument('--fps', help='Override frames per second of video and waveform as recorded in wave metadata', default=None, type=float)
    parser.add_argument('--window', help='Window size for fft HR calculation, in seconds', default=10, type=float)
    parser.add_argument('--moveCost', help='Parameter when using sssp algorithm', default=10, type=float)
    parser.add_argument('--plotPeaks', help='If run with --hrCalculator=peaks, location to save plotted peaks')
    parser.add_argument('--plotPeaksBounds', help='If run with --plotPeaks, set START and END bounds, in seconds (default: plot full waveform)', nargs=2, type=float, metavar=('START', 'END'))
    parser.add_argument('--lowHz', help='Low bound of frequency for HR', default=0.66666, type=float)
    parser.add_argument('--highHz', help='High bound of frequency for HR', default=3, type=float)
    parser.add_argument('--binWidth', help='Width of frequency bins in hz for fft calculation', default=0.001, type=float)
    parser.add_argument('--dequantize', action='store_true', help='Apply the dequantizing subroutine for fft and sssp')
    parser.add_argument('--deltaLimit', help='Limit change in HR; units of bpm/second (occurs before smooth if specified)', type=float)
    parser.add_argument('--smooth', help='Smooth using a sliding window; parameter is window width in seconds (occurs after deltaLimit if specified)', type=float)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    import numpy as np
    from rPPG.utils import hr
    from rPPG.utils import npio
    from pathlib import Path

    output = Path(args.output)
    if Path(args.waveform[0]).is_dir() and len(args.waveform) == 1:
        wavesMeta = {path: npio.loadMeta(path) for path in Path(args.waveform[0]).iterdir()}
        output.mkdir(exist_ok=True, parents=True)
    else:
        wavesMeta = {path: npio.loadMeta(path) for path in args.waveform}
    waves = {path: wm[0] for path, wm in wavesMeta.items()}
    meta = {path: wm[1] for path, wm in wavesMeta.items()}
    hrs = hr.calcHR(waves, args.fps if args.fps else list(meta.values())[0].fps(), hrCalculator=args.hrCalculator,
            lowHz = args.lowHz, highHz = args.highHz,
            window = args.window, dequantize = args.dequantize, binWidth = args.binWidth,
            moveCost = args.moveCost, deltaLimit = args.deltaLimit, smooth = args.smooth,
            skipFFT=not args.saveFFT)
    
    for waveNum, (path, (HR, fftOrPeaks)) in enumerate(hrs.items()):
        wave = waves[path]
        fps = args.fps if args.fps else meta[path].fps()
        if args.hrCalculator.startswith('peaks') and args.plotPeaks:
            peaks = fftOrPeaks
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title(f'Peaks detected')
            xMin = int(args.plotPeaksBounds[0] * fps) if args.plotPeaksBounds else 0
            xMax = int(args.plotPeaksBounds[1] * fps) if args.plotPeaksBounds else len(wave)
            plt.plot([x / fps for x in range(xMin, xMax)], wave[xMin:xMax])
            peaksBelow = np.array([p for p in peaks if p <= xMax and p >= xMin])
            plt.plot([p / fps for p in peaksBelow], wave[peaksBelow], 'x', markersize=10, markeredgewidth=5)
            plt.xlabel('Time (s)')
            plt.ylabel('Waveform')
            plt.savefig(args.plotPeaks if len(args.waveform) == 1 else str(waveNum) + args.plotPeaks)
            plt.close()
        elif args.saveFFT:
            npio.save(fftOrPeaks, meta[path], args.saveFFT if len(args.waveform) == 1 else str(waveNum) + args.saveFFT)
        print(f'Average for {path}: {sum(HR) / len(HR)}')
        savePath = output
        if len(args.waveform) > 1:
            savePath = output.parent / (str(waveNum) + output.name)
        if output.is_dir():
            savePath = output / path.name
        npio.save(HR, meta[path], savePath)
    
    if args.plotHR:
        import matplotlib.pyplot as plt
        plt.figure()
        if args.plotHR_title:
            plt.title(args.plotHR_title)
        else:
            plt.title(f'HR for {args.hrCalculator}')
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate (BPM)')
        for fname, (HR, _) in hrs.items():
            times = [float(i) / fps for i in range(len(HR))]
            plt.plot(times, HR, label=fname)
        if args.plotHR_ylims:
            plt.ylim(args.plotHR_ylims[0], args.plotHR_ylims[1])
        if len(args.waveform) > 1:
            plt.legend()
        plt.savefig(args.plotHR)
        plt.show()
