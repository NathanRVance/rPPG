#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Crop and scale a video based on bounding boxes')
parser.add_argument('boxes', help='Path to csv file where bounding boxes are saved')
parser.add_argument('video', help='Path to video to crop')
parser.add_argument('--intermediary', help='Path to intermediary video if size differs')
parser.add_argument('--label', help='Boxes label (uses all if not provided)')
parser.add_argument('--save-stem', help='Location to save stem.npz and stem.avi. Default: out', default='out')
parser.add_argument('--skip-zero-boxes', help='Skip boxes with zero size', action='store_true')
parser.add_argument('--debug', help='Save a debugging video as well', action='store_true')

argcomplete.autocomplete(parser)
args = parser.parse_args()

OUTW = 64
OUTH = 64

import csv
with open(args.boxes) as f:
    boxes = list(csv.reader(f))

if args.label:
    boxes = [b for b in boxes if b[0] == args.label]

import cv2
cap = cv2.VideoCapture(args.video)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

zoomX = 1.0
zoomY = 1.0
if args.intermediary:
    inter = cv2.VideoCapture(args.intermediary)
    interWidth = inter.get(cv2.CAP_PROP_FRAME_WIDTH)
    interHeight = inter.get(cv2.CAP_PROP_FRAME_HEIGHT)
    zoomX = width / interWidth
    zoomY = height / interHeight
    inter.release()

# Each box is [str(label), float(time), int(x1), int(y1), int(x2), int(y2)]
boxes = [[b[0], float(b[1]), int(int(b[2]) * zoomX), int(int(b[3]) * zoomY), int(int(b[4]) * zoomX), int(int(b[5]) * zoomY)] for b in boxes]

if args.skip_zero_boxes:
    boxes = [b for b in boxes if b[2] != b[4] and b[3] != b[5]]

import numpy as np
outArry = []
outVid = cv2.VideoWriter(args.save_stem + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (OUTW, OUTH))

if args.debug:
    outDebug = cv2.VideoWriter(args.save_stem + '-debug.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(OUTW * height / OUTH + width), int(height)))

# This may take a bit, so do a progress bar
from progress.bar import IncrementalBar
bar = IncrementalBar('Frames Processed', max=cap.get(cv2.CAP_PROP_FRAME_COUNT), suffix='%(index)d/%(max)d - %(eta)d s')

frameNum = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    time = frameNum / fps
    # Make boxes[0] the most recent timestamp without going over current time
    while len(boxes) >= 2 and boxes[1][1] <= time:
        boxes = boxes[1:]
    x1, x2 = sorted([boxes[0][2], boxes[0][4]])
    y1, y2 = sorted([boxes[0][3], boxes[0][5]])
    cropped = frame[y1:y2, x1:x2]
    if cropped.size < 1:
        resized = np.zeros((OUTH, OUTW, 3), dtype=cropped.dtype)
    else:
        resized = cv2.resize(cropped, (OUTH, OUTW), interpolation=cv2.INTER_CUBIC)
    outArry.append(resized)
    outVid.write(resized)
    if args.debug:
        # Draw box on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Resize "resized"
        scaledUp = cv2.resize(resized, (int(OUTW * height / OUTH), int(height)), interpolation=cv2.INTER_AREA)
        concatted = cv2.hconcat([frame, scaledUp])
        outDebug.write(concatted)
    bar.next()
    frameNum += 1

bar.finish()

# Save out array
np.savez_compressed(args.save_stem + '.npz', np.array(outArry))
