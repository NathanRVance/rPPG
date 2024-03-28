#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import numpy as np
from rPPG.utils import npio
from rPPG.utils.combineWaves import bandpass
from rPPG.utils import hr

def hilbert(wave, backend):
    if backend == 'scipy':
        from scipy.signal import hilbert
        return np.abs(hilbert(wave-np.mean(wave)))+np.mean(wave)
    elif backend == 'pytorch':
        import torch
        if not torch.is_tensor(wave):
            wave = torch.tensor(wave).cuda()
        N = wave.shape[-1]
        mean = torch.mean(wave, -1)
        wave -= mean[..., None]
        transforms = torch.fft.fft(wave, axis=-1)
        transforms[..., 1:N//2] *= 2.0
        transforms[..., N//2 + 1: N] = 0+0j
        return (torch.abs(torch.fft.ifft(transforms)) + mean[..., None]).double()
    else:
        raise ValueError(f'Unknown hiblert backend: {backend}')

def peaks(wave, fps, lowHz=0.66666, highHz=3, extrapolate=True):
    try:
        _, peaks = hr.calcHRPeaks(wave, fps, lowHz, highHz, backend='pyampd')
        if len(peaks) < 2:
            raise ValueError(f'Only obtained {len(peaks)} peaks')
    except ValueError:
        _, peaks = hr.calcHRPeaks(wave, fps, lowHz, highHz, backend='scipy')
    from scipy.interpolate import CubicSpline
    return CubicSpline(peaks, wave[peaks], bc_type='natural', extrapolate=extrapolate)(range(len(wave)))

def winmax(wave, fps, lowHz=0.66666, highHz=3):
    import torch
    if not torch.is_tensor(wave):
        wave = torch.tensor(wave)
    emulateBS = len(wave.shape) == 1
    if emulateBS: # Emulate batch size
        wave = wave[np.newaxis, :]
    fft = torch.abs(torch.fft.rfft(wave))
    N=fft.shape[-1]
    freqs = torch.fft.rfftfreq(2*N-1, 1/fps)
    freq_idcs = torch.logical_and(freqs >= lowHz, freqs <= highHz)
    freqs = freqs[freq_idcs]
    fft = fft[..., freq_idcs]
    domfreq = freqs[torch.argmax(fft, -1)] * .9
    window = torch.min((fps/domfreq).int())
    windows = wave.unfold(-1, window, 1) # Sliding window
    maxes = torch.max(windows, -1).values
    padded = torch.nn.functional.pad(maxes, (torch.floor(window/2).int(), torch.ceil(window/2).int()), mode='replicate')
    if emulateBS:
        padded = torch.squeeze(padded, 0)
    return padded

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Calculate the envelope for a signal', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('waveform', nargs='+', help='Path to the waveform to process')
    parser.add_argument('--output', help='Path to write envelope output (only for 1 waveform)')
    parser.add_argument('--bandpass', type=float, help='bandpass tolerance around HR')
    parser.add_argument('--method', nargs='+', choices=['hilbert', 'peaks', 'winmax'], default=['hilbert'], help='Envelope calculation method to use. If more than one provided, then applies in order of waveforms.')
    parser.add_argument('--hilbertBackend', choices=['scipy', 'pytorch'], default='scipy', help='Backend for the hilbert transform')
    parser.add_argument('--normalize', action='store_true', help='Normalize waveforms prior to envelope calculation and plotting')
    parser.add_argument('--scaleUp', action='store_true', help='Scale waves such that envelopes share an  average value with the largest one (only affects plotting)')
    parser.add_argument('--plot', help='Save a plot of the envelope')
    parser.add_argument('--labels', nargs='+', help='Labels for plotted waveforms (default is indexed)')
    parser.add_argument('--title', help='Title for the plot')
    parser.add_argument('--xlabel', help='X label for the plot', default='Time (s)')
    parser.add_argument('--ylabel', help='Y label for the plot', default='Amplitude')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    while len(args.waveform) > len(args.method):
        args.method.append(args.method[-1])
    
    partsToPlot = []

    for i, (fname, method) in enumerate(zip(args.waveform, args.method)):
        wave, meta = npio.loadMeta(fname)
        
        if args.normalize:
            wave -= np.mean(wave)
            wave /= np.std(wave)

        waveProcessed = wave
        if args.bandpass:
            HR = hr.calcHR({'a': wave}, meta.fps())['a'][0]
            waveProcessed = bandpass(wave, meta.fps(), cuttoffs=[max(1, (min(HR)-args.bandpass)/60), (max(HR)+args.bandpass)/60]).copy()
            # Bandpassing shifts it, so readjust
            waveProcessed += np.mean(wave)

        def getTopEnvelope(wave):
            if method == 'hilbert':
                envelope = hilbert(wave, args.hilbertBackend)
                if args.hilbertBackend == 'pytorch':
                    envelope = envelope.cpu().numpy()
                return envelope
            elif method == 'peaks':
                return peaks(wave, meta.fps(), extrapolate=False)
            elif method == 'winmax':
                envelope = winmax(wave, meta.fps()).cpu().numpy()
                return envelope
            else:
                raise ValueError(f'Unknown envelope method: {method}')

        envelope = [getTopEnvelope(waveProcessed), -getTopEnvelope(-waveProcessed)]
        if args.output:
            npio.save(envelope, meta, args.output)

        # Retain for plotting
        partsToPlot.append([wave, waveProcessed, np.array(envelope)])

    if args.scaleUp:
        target = [max(np.nanmedian(p2p[-1][i]) for p2p in partsToPlot) for i in [0, 1]]
        p2pNew = []
        for p2p in partsToPlot:
            src = [np.nanmedian(p2p[-1][i]) for i in [0, 1]]
            scale = (target[1]-target[0])/(src[1]-src[0])
            shift = target[0]-(scale*src[0])
            p2pNew.append([part*scale+shift for part in p2p])
        partsToPlot = p2pNew

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(args.title)
        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
        handles = {}
        for i, (wave, waveProcessed, envelope) in enumerate(partsToPlot):
            labelSuffix = '' if len(args.waveform) == 1 else f' {i}'
            if args.labels:
                labelSuffix = ' ' + args.labels[i]
            handles[f'signal{labelSuffix}'] = [plt.plot([i/meta.fps() for i in range(len(wave))], wave)[0]]
            if args.bandpass:
                handles[f'bandpassed{labelSuffix}'] = [plt.plot([i/meta.fps() for i in range(len(waveProcessed))], waveProcessed)[0]]
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            handles[f'envelope{labelSuffix}'] = [plt.plot([i/meta.fps() for i in range(len(e))], e, color=color)[0] for e in envelope]
            from matplotlib.legend_handler import HandlerTuple
            plt.legend([handle[0] for handle in handles.values()], handles.keys(), handler_map={tuple: HandlerTuple(ndivide=None)})

        plt.tight_layout()
        plt.savefig(args.plot)
        plt.show()
