#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Convert MMSE-HR database to same format as raw DDPM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('basedir', help='Base directory for MMSE-HR database')
parser.add_argument('out', help='Output directory for converted database')

argcomplete.autocomplete(parser)
args = parser.parse_args()

from pathlib import Path
import json
import csv
import cv2
import numpy as np

fps=25

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

for imgs, physdata in [('first 10 subjects 2D', 'first 10 subjects Phydata released/Phydata'), ('T10_T11_30Subjects', 'T10_T11_30PhyBPHRData')]:
    for subPath in (Path(args.basedir) / imgs).iterdir():
        subID = subPath.stem
        for taskPath in subPath.iterdir():
            taskID = taskPath.stem
            print(f'Processing subID: {subID}-{taskID}')
            subOut = out / f'{subID}-{taskID}'
            subOut.mkdir(exist_ok=True)
            physPath = Path(args.basedir) / physdata / subID / taskID
            with (physPath / 'Pulse Rate_BPM.txt').open() as f:
                bpm = [line for line in f.read().split('\n') if line]
            with (physPath / 'BP_mmHg.txt').open() as f:
                bp = [line for line in f.read().split('\n') if line]
            # Convert to csv: hr,o2,waveform,timestamp(seconds), but coopt o2 as bp.
            with (subOut / 'oximeterData.csv').open('w') as f:
                writer = csv.writer(f)
                for i, (hr, pressure) in enumerate(zip(bpm, bp)):
                    writer.writerow([float(hr), float(pressure), '', i/1000])
            # Now convert video
            outVid = None
            for fname in sorted(taskPath.glob('*.jpg')):
                #print(f'Loading frame from {fname}')
                frame = cv2.imread(str(fname))
                if not outVid:
                    h, w, _ = frame.shape
                    outVid = cv2.VideoWriter(str(subOut / 'compressed_RGBVideo.avi'), cv2.VideoWriter_fourcc('F','F','V','1'), fps, (w, h))
                outVid.write(frame)
