#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Convert PURE database to same format as raw DDPM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('basedir', help='Base directory for PURE database')
parser.add_argument('out', help='Output directory for converted database')
parser.add_argument('--check', help='Do not output video but rather check that existing video matches input', action='store_true')

argcomplete.autocomplete(parser)
args = parser.parse_args()

from pathlib import Path
import json
import csv
import cv2
import numpy as np

fps=30

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

for subPath in (Path(args.basedir) / 'data').iterdir():
    subID = subPath.stem
    print(f'Processing subID: {subID}')
    subOut = out / subID
    subOut.mkdir(exist_ok=True)
    with open(subPath / (subID+'.json')) as f:
        oximeterDat = json.load(f)
    # Convert to csv: hr,o2,waveform,timestamp(seconds)
    # Convert from keys: "pulseRate", "o2saturation", "waveform", "Timestamp" (ns)
    minFrameTS = oximeterDat['/Image'][0]['Timestamp']
    with (subOut / 'oximeterData.csv').open('w') as f:
        writer = csv.writer(f)
        for frame in oximeterDat['/FullPackage']:
            writer.writerow([frame['Value']['pulseRate'], frame['Value']['o2saturation'], frame['Value']['waveform'], (frame['Timestamp']-minFrameTS)*1e-9])
    # Now convert video
    inVid = None if not args.check else cv2.VideoCapture(str(subOut / 'compressed_RGBVideo.avi'))
    outVid = None
    for fname in sorted((subPath / subID).glob('*.png')):
        #print(f'Loading frame from {fname}')
        frame = cv2.imread(str(fname))
        if args.check:
            _, inFrame = inVid.read()
            if np.array_equal(frame, inFrame):
                print('Frames match')
            else:
                print('Frames do not match')
        if not outVid and not args.check:
            h, w, _ = frame.shape
            outVid = cv2.VideoWriter(str(subOut / 'compressed_RGBVideo.avi'), cv2.VideoWriter_fourcc('F','F','V','1'), fps, (w, h))
        if outVid:
            outVid.write(frame)
