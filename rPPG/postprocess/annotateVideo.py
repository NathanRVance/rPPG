#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from rPPG.utils import vidcap
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rPPG.utils import npio
from rPPG.utils import bboxUtils
import sys

def plotSimple(title, xVals, yVals, xLabel, yLabel, h_px, w_px):
    fig = plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.title(title)
    plt.plot(xVals, yVals)
    plt.gca().set_ylabel(yLabel)
    plt.gca().set_xlabel(xLabel)
    plt.tight_layout()
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    # Shrink down
    img = cv2.resize(img, (h_px, w_px), interpolation=cv2.INTER_AREA)
    plt.close('all')
    return img

def plot(title, yVals, yLabel, xi_ctr, h_px, w_px, plotWidth, fps):
    fig = plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.title(title)
    nones = 0 if None not in yVals else int(np.argwhere(yVals != None)[0][0])
    xi_min = max(nones, int((xi_ctr/fps - plotWidth/2)*fps))
    xi_max = min(len(yVals), int((xi_ctr/fps + plotWidth/2)*fps))
    x = [i / fps for i in range(xi_min, xi_max)]
    plt.plot(x, yVals[xi_min:xi_max])
    plt.gca().set_ylim(min(yVals[nones:]), max(yVals[nones:]))
    if len(x) == 0:
        plt.gca().set_xlim(nones/fps-plotWidth, nones/fps)
    plt.gca().set_ylabel(yLabel)
    plt.gca().set_xlabel('Time (s)')
    plt.axvline(xi_ctr/fps)
    plt.gca().set_xticklabels([])
    plt.tight_layout()
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    #img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # Shrink down
    img = cv2.resize(img, (h_px, w_px), interpolation=cv2.INTER_AREA)
    plt.close('all')
    return img

def drawBBox(frame, box, index=1):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = bboxUtils.shift(box, w, h)
    color = [0, 0, 0]
    color[index%len(color)] = 255
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame

def rotate_image(image, angle):
    import imutils
    return imutils.rotate_bound(image, angle=angle)

def annotateVideo(video, output, waveform=[], hrData=[], fftData=[], bboxes=[], outHeight=720, plotWidth=10, maxPlotsInColumn=3, maxPlotsInRow=2, waveformNames=[], hrNames=[], fftNames=[], plotByIndex=False, rotation=0):
    output = str(output)
    with vidcap.Reader(video, 'Boxes Processed') as cap:
        scale = float(outHeight) / cap.height()
        outDims = (int(scale*cap.width()), int(scale*cap.height()))
        if rotation != 0 and rotation != 180:
            outDims = (outDims[1], outDims[0])
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M','J','P','G'), cap.fps, outDims)

        waves = [npio.load(wave) for wave in waveform]
        HRs = [npio.load(hrFile) for hrFile in hrData]
        fftData = [npio.load(fftFile) for fftFile in fftData]
        bboxes = [npio.load(bbox) for bbox in bboxes]
        if waves:
            wave = waves[-1]
            offset = ((len(wave)-len(HRs[-1]))//2)
            # HRs and fftData should be centered on the wave
            HRs = [np.append([None] * offset, HR) for HR in HRs]
            fftData = [np.append(np.full((offset, FFT.shape[1], FFT.shape[2]), None), FFT, axis=0) for FFT in fftData]

        for frameNum, frame in enumerate(cap):
            time = cap.time
            for index in range(len(bboxes)):
                while len(bboxes[index]) >= 2 and bboxes[index][1][0] <= time:
                    bboxes[index] = bboxes[index][1:]
                box = [int(b) for b in bboxes[index][0][1:]]
                frame = drawBBox(frame, box, index)
            if rotation != 0:
                frame = rotate_image(frame, rotation)
            # Scale frame to out dimensions
            frame = cv2.resize(frame, outDims, interpolation=cv2.INTER_AREA)
            plotDims = int(outHeight / min(maxPlotsInColumn, (len(HRs)+len(fftData)+len(waves)+1)))
            plots = {'wave': [], 'hr': [], 'fft': []}
            # Plot wave
            def getTitles(prefix, data, fnames, names):
                for i, (dat, name) in enumerate(zip(data, fnames)):
                    title = f'{prefix} ({name})'
                    if names and len(names) > i:
                        title = f'{prefix} ({names[i]})'
                    elif len(fnames) == 1:
                        title = prefix
                    yield title, dat
            for title, wave in getTitles('Wave', waves, waveform, waveformNames):
                plots['wave'].append(plot(title, wave, 'Waveform', frameNum, plotDims, plotDims, plotWidth, cap.fps))
            for title, HR in getTitles('HR', HRs, hrData, hrNames):
                plots['hr'].append(plot(title, HR, 'BPM', frameNum, plotDims, plotDims, plotWidth, cap.fps))
            for title, fft in getTitles('FFT', fftData, fftData, fftNames):
                if frameNum < len(fft) and None not in fft[frameNum]:
                    plots['fft'].append(plotSimple(title, [n*60 for n in fft[frameNum][0]], fft[frameNum][1], 'Frequency (BPM)', 'Density', plotDims, plotDims))
            # Place them top to bottom
            imgs = [v for pv in plots.values() for v in pv]
            if plotByIndex:
                imgs = []
                for i in range(max(len(p) for p in plots.values())):
                    for p in plots.values():
                        if i < len(p):
                            imgs.append(p[i])
            for i, img in enumerate(imgs):
                vOff = (i % maxPlotsInColumn) * plotDims
                hOff = (i // maxPlotsInColumn)
                if hOff < maxPlotsInRow:
                    hOff *= plotDims
                else:
                    hOff = outDims[0] - ((hOff - maxPlotsInRow + 1) * plotDims)
                frame[vOff:vOff+plotDims, hOff:hOff+plotDims] = img
            out.write(frame)

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Annotate a video with the waveform and heart rate', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', help='Original video to annotate')
    parser.add_argument('output', help='Path to save annotated AVI video')
    parser.add_argument('--rotation', type=int, default=0, help='Degrees to rotate video and bboxes')
    parser.add_argument('--waveform', nargs='+', help='npz formatted waveform', default=[])
    parser.add_argument('--waveformNames', nargs='+', help='Names of waveforms')
    parser.add_argument('--hrData', nargs='+', help='HR data to plot', default=[])
    parser.add_argument('--hrNames', nargs='+', help='Names of hr')
    parser.add_argument('--fftData', nargs='+', help='FFT data to plot', default=[])
    parser.add_argument('--fftNames', nargs='+', help='Names of fft')
    parser.add_argument('--bboxes', nargs='+', help='Bounding boxes to annotate', default=[])
    parser.add_argument('--outHeight', help='Height of output video', type=int, default=720)
    parser.add_argument('--plotWidth', help='Width of plotted HR data in seconds', type=float, default=10)
    parser.add_argument('--maxPlotsInColumn', help='Maximum number of plots to place in a column', type=int, default=3)
    parser.add_argument('--maxPlotsInRow', help='Maximum number of plots to place in a row before jumping to the other side of the video', type=int, default=2)
    parser.add_argument('--plotByIndex', action='store_true', help='Place plots of same index near each other. Default places same type near each other.')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    annotateVideo(args.video, args.output, args.waveform, args.hrData, args.fftData, args.bboxes, args.outHeight, args.plotWidth, args.maxPlotsInColumn, args.maxPlotsInRow, args.waveformNames, args.hrNames, args.fftNames, args.plotByIndex, args.rotation)
