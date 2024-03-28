#!/usr/bin/env python3

import cv2
import time

class Reader():
    def __init__(self, fname, bartext='', interpolate=True):
        self.fname = str(fname)
        self.fdelta = 0
        self.frameCurr = None
        self.interpolate = interpolate
        if self.fname == '0':
            # Interpolation doesn't work with webcams (dunno why)
            self.interpolate = False
        self.bartext = bartext
        self.bar = None

    def height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def __enter__(self):
        if self.fname == '0': # Webcam
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FORMAT, -1) # raw
        else:
            self.cap = cv2.VideoCapture(str(self.fname))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fname == '0': # Use wall time
            self.timeStart = time.time()
            self.time = self.timeStart
            # Also, adjust fps downward
            self.fps /= 2
        else:
            self.time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        _, self.frame = self.cap.read()
        if self.bartext:
            from progress.bar import IncrementalBar
            self.bar = IncrementalBar(self.bartext, max=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))+1)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.bar:
            self.bar.next()
        if self.fdelta <= 0:
            ret, self.frameCurr = self.cap.read()
            while ret and self.frameCurr is None:
                ret, self.frameCurr = self.cap.read()
            if self.fname == '0': # Use wall time
                timeCur = time.time() - self.timeStart
            else:
                timeCur = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self.interpolate:
                self.fdelta = int(round((timeCur-self.time)*self.fps))
            else:
                self.fdelta = 1
            self.time = timeCur
            if not ret: #or self.fdelta == 0:
                if self.frame is not None:
                    ret = self.frame
                    self.frame = None
                    return ret
                self.bar.finish()
                raise StopIteration
        #print(f'fdelta: {self.fdelta}')
        ret = self.frame
        if self.fdelta == 1:
            self.frame = self.frameCurr
        elif self.fdelta > 1:
            # Interpolate frames
            self.frame = self.frame + (self.frameCurr-self.frame)/self.fdelta
            self.frame = self.frame.astype(self.frameCurr.dtype)
        self.fdelta -= 1
        return ret

if __name__ == '__main__':
    import sys
    import hashlib
    import cv2
    with Reader(sys.argv[-1], 'stuff') as cap:
        for frameNum, frame in enumerate(cap):
            cv2.imshow(sys.argv[-1], frame)
            if cv2.waitKey(1) == ord('q'):
                break
