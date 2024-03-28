import numpy as np
import random
import cv2

def procBoxes(boxes, scale=1) -> list:
    """Performs basic bbox processing (squaring and scaling)
    
    Args:
      boxes: A list of [[timestamp, x1, y1, x2, y2], ...]
      scale: Factor to scale all boxes (default 1)

    Returns:
      list: Processed boxes
    """
    # Remove null boxes
    boxes = [b for b in boxes if all(x1 < x2 and y1 < y2 for x1, y1, x2, y2 in zip(b[1::4], b[2::4], b[3::4], b[4::4]))]

    if scale != 1:
        # Scale boxes
        boxes = [[b[0], int(b[1] / scale), int(b[2] / scale), int(b[3] / scale), int(b[4] / scale)] for b in boxes]

    # Square them up
    boxes = [[b[0]] + square(b[1:]) for b in boxes]
    return boxes

def square(box: list) -> list:
    """Make a bounding box square by increasing the shortest dimension

    Args:
      box: The bounding box, formatted [x1, y1, x2, y2]

    Returns:
      list: The squared box
    """
    coords1, coords2 = [0, 2], [1, 3]
    if box[coords2[1]] - box[coords2[0]] > box[coords1[1]] - box[coords1[0]]:
        coords1, coords2 = coords2, coords1
    # coords1 > coords2
    d = (box[coords1[1]] - box[coords1[0]]) - (box[coords2[1]] - box[coords2[0]])
    box[coords2[0]] -= int(d/2)
    box[coords2[1]] += int(d/2) + (d % 2)
    return box

def shift(box: list, w: int, h: int) -> list:
    """Shift a bounding box to be within w,h bounds

    Args:
      box: The bounding box, formatted [x1, y1, x2, y2]
      w: The width of the frame (i.e., maximum x value)
      h: The height of the frame (i.e., maximum y value)

    Returns:
      list: The shifted box
    """
    bounds = [[0, int(w)], [0, int(h)]]
    coords = [[0, 2], [1, 3]]
    for b, c in zip(bounds, coords):
        s = 0
        if box[c[0]] < b[0]:
            s = -box[c[0]]
        elif box[c[1]] > b[1]:
            s = b[1] - box[c[1]]
        box[c[0]] += s
        box[c[1]] += s
    return box

def landmarks2bbox(lmks: dict, pads: list = [-0.05, -0.3, 0.05, 0.05]) -> list:
    """Define a bounding box from OpenFace/MediaPipe landmarks

    Args:
      lmrks: The face landmarks, formatted np.array([[x1, y1], [x2, y2], ...])
      pads: Percent padding that is applied to [xmin, ymin, xmax, ymax] (origin is upper left)

    Returns:
      list: The bounding box from landmarks as [x1, y1, x2, y2]
    """
    box = [min([l[0] for l in lmks]), min([l[1] for l in lmks]),
            max([l[0] for l in lmks]), max([l[1] for l in lmks])]
    W, H = box[2]-box[0], box[3]-box[1]
    pads = [p * wh for p, wh in zip(pads, [W, H, W, H])]
    box = [int(b+p) for b, p in zip(box, pads)]
    box = square(box)
    return [int(b) for b in box]

def getFaceMask(frame, bbox, landmarks):
    """Obtain a face mask for a frame.

    We do convex hull containing lmks 0-16, 27-35, and extend edges upward so that lmk 28 is centered vertically.
    We then mask out the eyes and mouth.

    Args:
      frame: The image after cropping but before downsampling
      bbox: The bounding box used for cropping
      landmarks: The openface landmarks as a dictionary

    Returns:
      np.array: Mask of the face
    """
    mask = np.zeros(frame.shape[:2], dtype='uint8')
    def getLmk(lmkID):
        return [int(float(landmarks[f' {dim}_{lmkID}'])) for dim in ['x', 'y']]
    hullPts = [getLmk(lmk) for lmk in list(range(17)) + list(range(27, 36))]
    # Add two points to center lmk 30 vertically
    cnt = getLmk(28)
    yToCenter = cnt[1] - (max(p[1] for p in hullPts) - cnt[1])
    hullPts.append([min(p[0] for p in hullPts), yToCenter])
    hullPts.append([max(p[0] for p in hullPts), yToCenter])
    hullPts = cv2.convexHull(np.array(hullPts))
    cv2.fillPoly(mask, pts=[hullPts], color=255)
    # Now mask back out eyes and mouth
    for ptsList in [range(36, 42), range(42, 48), range(48, 60)]:
        pts = cv2.convexHull(np.array([getLmk(pt) for pt in ptsList]))
        cv2.fillPoly(mask, pts=[pts], color=0)
    return mask

class FaceCropperBase(object):
    def __init__(self, resizeX, resizeY):
        self.resizeX = resizeX
        self.resizeY = resizeY

    def getRegion(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]

    def resizeRegion(self, frame):
        if frame.shape[0] <= 1 or frame.shape[1] <= 1:
            return np.zeros((self.resizeX, self.resizeY, 3), dtype=frame.dtype)
        else:
            return cv2.resize(frame, (self.resizeX, self.resizeY), interpolation=cv2.INTER_CUBIC)

    def cropFace(self, frame, bbox, landmarks):
        h, w, _ = frame.shape
        bbox = shift(bbox[:4], w, h)
        return self.getRegion(frame, bbox)

    def cropAndResize(self, frame, bbox, landmarks):
        cropped = self.cropFace(frame, bbox, landmarks)
        return self.resizeRegion(cropped)

class FaceCropperPlain(FaceCropperBase):
    pass # Identical to FaceCropperBase

class FaceCropperSubregions(FaceCropperBase):
    def cropFace(self, frame, bbox, landmarks):
        # We use top half for forehead, and bottom half (split l/r) for cheeks
        forehead = self.getRegion(frame, bbox[4:8])
        lCheek = self.getRegion(frame, bbox[8:12])
        rCheek = self.getRegion(frame, bbox[12:16])
        croppedH = forehead.shape[0] * 2
        croppedW = forehead.shape[1]
        cropped = np.zeros((croppedH, croppedW, 3), np.uint8)
        cropped[0:forehead.shape[0], 0:croppedW] = forehead
        cropped[forehead.shape[0]:croppedH, 0:croppedW//2] = cv2.resize(lCheek, (croppedW//2, forehead.shape[0]), interpolation=cv2.INTER_CUBIC)
        import math
        cropped[forehead.shape[0]:croppedH, croppedW//2:croppedW] = cv2.resize(rCheek, (math.ceil(croppedW/2), forehead.shape[0]), interpolation=cv2.INTER_CUBIC)
        return cropped

class FaceCropperShuffleAll(FaceCropperBase):
    def __init__(self, resizeX, resizeY, preserve):
        super().__init__(resizeX, resizeY)
        import random
        self.shuffleOrder = list(range(self.resizeX*self.resizeY))
        random.shuffle(self.shuffleOrder)
        self.preserve = preserve

    def cropAndResize(self, frame, bbox, landmarks):
        frame = super().cropAndResize(frame, bbox, landmarks)
        if self.preserve:
            shuffleOrder = self.shuffleOrder
        else:
            shuffleOrder = list(range(self.resizeX*self.resizeY))
            random.shuffle(shuffleOrder)
        pixels = [frame[y, x] for x in range(self.resizeX) for y in range(self.resizeY)]
        withOrder = [[order, px] for order, px in zip(shuffleOrder, pixels)]
        withOrder = sorted(withOrder, key=lambda wo: wo[0])
        pixels = [wo[1] for wo in withOrder]
        for x in range(self.resizeX):
            for y in range(self.resizeY):
                frame[y, x] = pixels[0]
                pixels = pixels[1:]
        return frame

class FaceCropperShufflePreserveBackground(FaceCropperBase):
    def __init__(self, resizeX, resizeY, preserve):
        super().__init__(resizeX, resizeY)
        import random
        self.shuffleOrder = list(range(self.resizeX*self.resizeY))
        random.shuffle(self.shuffleOrder)
        self.preserve = preserve

    def cropAndResize(self, frame, bbox, landmarks):
        h, w, _ = frame.shape
        bbox = shift(bbox[:4], w, h)
        mask = getFaceMask(frame, bbox, landmarks)
        frame = self.cropFace(frame, bbox, landmarks)
        mask = self.cropFace(mask, bbox, landmarks)
        frame = self.resizeRegion(frame)
        mask = self.resizeRegion(mask)
        # Should still be "binary"
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        maskPixels = [frame[y, x] for x in range(self.resizeX) for y in range(self.resizeY) if mask[y, x]]
        if self.preserve:
            # Rearrange pixels in the mask based on self.shuffleOrder
            maskPixelsWithOrder = [[order, px] for order, px in zip(self.shuffleOrder[:len(maskPixels)], maskPixels)]
            maskPixelsWithOrder = sorted(maskPixelsWithOrder, key=lambda mpwo: mpwo[0])
            maskPixels = [mpwo[1] for mpwo in maskPixelsWithOrder]
        else:
            # Instead, let's just shuffle.
            random.shuffle(maskPixels)
        for x in range(self.resizeX):
            for y in range(self.resizeY):
                if mask[y, x]:
                    frame[y, x] = maskPixels[0]
                    maskPixels = maskPixels[1:]
        return frame
