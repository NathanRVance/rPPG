#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Convert UBFC-rPPG database to same format as raw DDPM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('basedir', help='Base directory for UBFC-rPPG database')
parser.add_argument('out', help='Output directory for converted database')

argcomplete.autocomplete(parser)
args = parser.parse_args()

from pathlib import Path
import json
import csv
import cv2
import numpy as np

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

for subPath in (Path(args.basedir) / 'DATASET_2').iterdir():
    subID = subPath.stem
    print(f'Processing subID: {subID}')
    subOut = out / subID
    subOut.mkdir(exist_ok=True)
    oximeterDat = np.loadtxt(subPath / 'ground_truth.txt')
    # Convert to csv: hr,o2,waveform,timestamp(seconds)
    # Convert from [3,N] array: waveform, hr, timestamp
    with (subOut / 'oximeterData.csv').open('w') as f:
        writer = csv.writer(f)
        for waveform, hr, timestamp in zip(*oximeterDat):
            writer.writerow([hr, -1, waveform, timestamp])
    # Now symlink video
    (subOut / 'compressed_RGBVideo.avi').symlink_to(subPath / 'vid.avi')
    
