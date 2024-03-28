#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import mediapipe as mp
from rPPG.utils import mediapipe
from rPPG.utils import npio
from rPPG.utils import bboxUtils
from rPPG.utils import vidcap

def calcMP(fname, drawBar=True):
    boxes = [] # [[float(time), int(x1), int(y1), int(x2), int(y2)], ...]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_detection:
        with vidcap.Reader(fname, 'Boxes Processed' if drawBar else '') as cap:
            for frame in cap:
                box = calcBoxSingle(frame, cap.time, face_detection, False)
                if box:
                    boxes.append(box)
    return boxes

def calcBoxSingle(frame, time, face_detection, postprocess=False):
    bounds = mediapipe.calcMPFrame(frame, face_detection)
    if bounds:
        box = [time] + bounds
        if postprocess:
            box = bboxUtils.procBoxes([box])[0]
        return box
    return None

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Extract bounding boxes using mediapipe', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('output', help='Path to output npz file')
    parser.add_argument('--skip-postprocess', action='store_true', help='Skip standard postprocessing steps such as making the box square')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    boxes = calcMP(args.input)
    if not args.skip_postprocess:
        boxes = bboxUtils.procBoxes(boxes)

    npio.save(boxes, {}, args.output)

