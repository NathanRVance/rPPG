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
import tempfile
import zipfile

fps=25

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


for subPath in (Path(args.basedir) / 'Physiology').iterdir():
    subID = subPath.stem
    for taskPath in subPath.iterdir():
        taskID = taskPath.stem
        print(f'Processing subID: {subID}-{taskID}')
        subOut = out / f'{subID}-{taskID}'
        subOut.mkdir(exist_ok=True)
        if not (subOut / 'oximeterData.csv').exists():
            with (taskPath / 'Pulse Rate_BPM.txt').open() as f:
                bpm = [line for line in f.read().split('\n') if line]
            with (taskPath / 'BP_mmHg.txt').open() as f:
                bp = [line for line in f.read().split('\n') if line]
            with (taskPath / 'Respiration Rate_BPM.txt').open() as f:
                rr = [line for line in f.read().split('\n') if line]
            # Convert to csv: standard is hr,o2,waveform,timestamp(seconds),
            # but instead do              hr,rr,bp,      timestamp(seconds)
            with (subOut / 'oximeterData.csv').open('w') as f:
                writer = csv.writer(f)
                for i, (hr, pressure, resp) in enumerate(zip(bpm, bp, rr)):
                    writer.writerow([float(hr), float(resp), float(pressure), i/1000])
        # Check if video already exists
        if (subOut / 'compressed_RGBVideo.avi').exists():
            continue
        # Now convert video
        outVid = None
        # Extract vid to a tmp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            z = zipfile.ZipFile(Path(args.basedir) / '2D+3D' / f'{subID}.zip')
            toExtract = [fname for fname in z.namelist() if Path(fname).parent.name == taskID and Path(fname).suffix == '.jpg']
            z.extractall(path=tmpdir, members=toExtract)
            for fname in sorted((Path(tmpdir)/subID/taskID).glob('*.jpg')):
                #print(f'Loading frame from {fname}')
                frame = cv2.imread(str(fname))
                if not outVid:
                    h, w, _ = frame.shape
                    outVid = cv2.VideoWriter(str(subOut / 'compressed_RGBVideo.avi'), cv2.VideoWriter_fourcc('F','F','V','1'), fps, (w, h))
                outVid.write(frame)
