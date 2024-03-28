#!/usr/bin/env python3
import cv2
import mediapipe as mp
from rPPG.utils  import bboxUtils
import numpy as np

## This is a mapping from Mediapipe's FaceMesh coordinates to OpenFace's coordinates
CANONICAL_LMRKS = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

def calcMPFrame(frame, face_detection=None):
    def _calcMPFrame(frame, face_detection):
        if np.shape(frame) == ():
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, mark the image as not writeable to pass by reference
        frame.flags.writeable = False
        results = face_detection.process(frame)
        if not results.multi_face_landmarks:
            return None
        h, w, _ = frame.shape
        lmks = [[int(results.multi_face_landmarks[0].landmark[i].x*w), int(results.multi_face_landmarks[0].landmark[i].y*h)] for i in CANONICAL_LMRKS]
        return bboxUtils.landmarks2bbox(lmks)
    if not face_detection:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_detection:
            return _calcMPFrame(frame, face_detection)
    return _calcMPFrame(frame, face_detection)
