#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import numpy as np

def calcSignals(video, bboxes=None, drawBar=True):
    from rPPG.utils import vidcap
    def _calcSignals(video, face_detection, bboxes, drawBar):
        signals = []
        with vidcap.Reader(video, 'Frames Processed' if drawBar else None) as cap:
            for frameNum, frame in enumerate(cap):
                if bboxes is not None:
                    time = frameNum / cap.fps
                    while len(bboxes) >= 2 and bboxes[1][0] <= time:
                        bboxes = bboxes[1:]
                    bbox = bboxes[0][1:]
                else:
                    bboxNew = mediapipe.calcMPFrame(frame, face_detection)
                    if bboxNew:
                        bbox = bboxNew
                sigs = calcSignalsFrame(frame, bbox)
                if not np.any(sigs) and len(signals) > 0:
                    sigs = signals[-1]
                signals.append(sigs)
        return np.vstack(signals)
    face_detection = None
    if bboxes is None or len(bboxes) == 0:
        from rPPG.utils import mediapipe
        import mediapipe as mp
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_detection:
            return _calcSignals(video, face_detection, None, drawBar)
    else:
        return _calcSignals(video, None, bboxes, drawBar)

def calcSignalsFrame(frame, bbox):
    # Get spatial average within bbox
    x1,y1,x2,y2 = [int(b) for b in bbox]
    subframe = frame[y1:y2, x1:x2]
    if subframe.size > 0:
        return np.mean(subframe, axis=(0,1))
    else:
        return np.array([0, 0, 0])

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Compute CHROM and POS waveforms.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', help='Path to the input video')
    parser.add_argument('output', help='Path to output face spatial averages from video')
    parser.add_argument('--boxes', help='Path to saved bounding boxes')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    from rPPG.utils import metadata
    from rPPG.utils import npio

    meta = metadata.Metadata.fromVideo(args.video)
    signals = calcSignals(args.video, None if not args.boxes else npio.load(args.boxes))
    npio.save(signals, meta, args.output)

