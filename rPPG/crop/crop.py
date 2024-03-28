#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from rPPG.utils import vidcap
from rPPG.utils import metadata
import cv2
import numpy as np
from rPPG.utils import npio
from pathlib import Path
from rPPG.utils import bboxUtils

def crop(fname, boxeses, meta, output, average=[0], skip=[0], dims=64, faceCropper=None, landmarks=None, debugFname=None, debugScale=.25, drawBar=True):
    with vidcap.Reader(fname, 'Frames Cropped' if drawBar else '') as cap:
        if faceCropper is None:
            faceCropper = bboxUtils.FaceCropperPlain(dims, dims)
        if debugFname:
            debugWidth = int(debugScale * (cap.width() + cap.height()))
            debugHeight = int(debugScale * cap.height())
            outDebug = cv2.VideoWriter(debugFname, cv2.VideoWriter_fourcc('M','J','P','G'), cap.fps, (debugWidth, debugHeight))
        accumulators = {avg: [] for avg in average}
        outArrys = {avg: [[] for b in boxeses] for avg in average}
        lmks = None
        for frameNum, frame in enumerate(cap):
            time = frameNum / cap.fps
            # For each boxes in boxeses, make boxes[0] the most recent timestamp without going over current time
            for i in range(len(boxeses)):
                while len(boxeses[i]) >= 2 and boxeses[i][1][0] <= time:
                    boxeses[i] = boxeses[i][1:]
            if landmarks:
                # Same for landmarks
                while len(landmarks) >= 2 and float(landmarks[0][' timestamp']) <= time:
                    landmarks = landmarks[1:]
                lmks = landmarks[0]
            for avg, accumulator in accumulators.items():
                accumulator.append(frame)
                if len(accumulator) == avg+1:
                    avgFrame = np.mean(accumulator, axis=0).astype(np.uint8)
                    accumulator.clear()
                    for boxes, outArry in zip(boxeses, outArrys[avg]):
                        resized = cropSingle(avgFrame, boxes[0], lmks, faceCropper)
                        outArry.append(resized)
                    if debugFname:
                        writeDebug(frame, resized, [int(b) for b in boxeses[-1][1:]], lmks, outDebug, debugWidth, debugHeight)
    # Save data in outArrys
    # Input is {avg: [arry-by-box]}
    output = Path(output)
    for avg, arrys in outArrys.items():
        prefix1 = '' if len(outArrys.keys()) == 1 else f'a{avg}-'
        meta2 = meta.makeSkipped(avg)
        for b, outArry in enumerate(arrys):
            prefix2 = '' if len(arrys) == 1 else f'b{b}-'
            for sk in skip:
                prefix3 = '' if len(skip) == 1 else f's{sk}-'
                prefix = prefix1+prefix2+prefix3
                npio.save(np.array(outArry[::sk+1]), meta2.makeSkipped(sk), output.parent / (prefix + output.name))

def cropSingle(frame, bbox, lmks, faceCropper):
    if len(bbox) > 4: # We don't want time
        bbox = bbox[1:]
    return faceCropper.cropAndResize(frame, [int(b) for b in bbox], lmks)

def writeDebug(frame, resized, box, lmks, outDebug, debugWidth, debugHeight):
    # Draw box on frame
    debugFrame = frame.copy()
    for x1, y1, x2, y2 in zip(box[0::4], box[1::4], box[2::4], box[3::4]):
        cv2.rectangle(debugFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if lmks:
        # Also draw landmarks
        for lmk in range(60):
            coords = [int(float(lmks[f' {dim}_{lmk}'])) for dim in ['x', 'y']]
            cv2.circle(debugFrame, coords, 4, (0, 0, 255), -1)
    # Resize "resized"
    scaledUp = cv2.resize(resized, (debugHeight, debugHeight), interpolation=cv2.INTER_AREA)
    scaledDown = cv2.resize(debugFrame, (debugWidth-debugHeight, debugHeight), interpolation=cv2.INTER_CUBIC)
    concatted = cv2.hconcat([scaledDown, scaledUp])
    outDebug.write(concatted)

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Crop a video based on bounding boxes', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', help='Video to crop')
    parser.add_argument('boxes', nargs='+', help='npz formatted bounding boxes. Multiple parameters result in "bX-" prepended to output file name, where X is the index of the bounding box.')
    parser.add_argument('output', help='Path to output npz file')
    parser.add_argument('--dims', help='Crop and scale into a dims X dims sized array', default=64, type=int)
    parser.add_argument('--skip', nargs='+', help='Frames to "skip" when traversing the video and boxes, used to obtain lower FPS output. When skip=0, iterates over the video normally. Multiple parameters result in "sX-" prepended to the output file name, where X is the skip value.', default=[0], type=int)
    parser.add_argument('--average', nargs='+', type=int, help='Frames to "average" when traversing the video and boxes, used to obtain lower FPS output. When average=0, iterates over the video normally. Multiple parameters result in "aX-" prepended to the output file name, where X is the average value.', default=[0])
    parser.add_argument('--subregions', help='Use subregions when cropping', action='store_true')
    parser.add_argument('--debug', nargs='?', help='Generate a debugging video as well. Optionally accepts a file name. By default outputs to OUTPUT-debug.avi', const=True, default=False)
    parser.add_argument('--debugScale', type=float, help='Scale factor when generating debug video', default=0.25)
    parser.add_argument('--landmarks', help='csv openface landmarks (use with --debug to annotate, or provide for --shuffleFace)')
    parser.add_argument('--shuffleFace', choices=['shuffleAll', 'preserveBackground'], help='Shuffle facial features (requires --landmarks)')
    parser.add_argument('--shufflePreserveOrder', help='Shuffle once and preserve order throughout video', action='store_true')
    parser.add_argument('--embedGt', help='Embed gt in metadata undergoing fps conversion as in cvtData.py')
    parser.add_argument('--embedMask', help='Embed mask in metadata undergoing fps conversion as in cvtData.py')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    meta = metadata.Metadata.fromVideo(args.video)
    for key, fname in [['gt', args.embedGt], ['mask', args.embedMask]]:
        if not fname:
            continue
        meta.embedMetadata(key, fname)

    lmks = None
    if args.landmarks:
        import csv
        with open(args.landmarks) as f:
            lmks = list(csv.DictReader(f))

    if args.subregions:
        faceCropper = bboxUtils.FaceCropperSubregions(args.dims, args.dims)
    elif args.shuffleFace:
        if not args.landmarks:
            print('ERROR: Must provide --landmarks with --shuffleFace')
            exit(1)
        if args.shuffleFace == 'shuffleAll':
            faceCropper = bboxUtils.FaceCropperShuffleAll(args.dims, args.dims, args.shufflePreserveOrder)
        elif args.shuffleFace == 'preserveBackground':
            faceCropper = bboxUtils.FaceCropperShufflePreserveBackground(args.dims, args.dims, args.shufflePreserveOrder)
        else:
            print(f'ERROR: Unimplemented option {args.shuffleFace}')
            exit(1)
    else:
        faceCropper = bboxUtils.FaceCropperPlain(args.dims, args.dims)

    debugFname = args.output + '-debug.avi' if args.debug == True else args.debug

    crop(args.video, [npio.load(b) for b in args.boxes], meta, args.output, average=args.average, skip=args.skip, dims=args.dims, faceCropper=faceCropper, landmarks=lmks, debugFname=debugFname, debugScale=args.debugScale, drawBar=True)
