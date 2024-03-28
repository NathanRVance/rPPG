#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from rPPG.utils import bboxUtils    

def calcSubregions(header: list, lmks: list, box: list) -> list:
    """Calculate openface subregions

    3 regions are calculated:
      Forehead: Region above and between edges of eyebrows (17-26)
      Cheeks: Regions below eyes (37-47), between edges of face (0-4, 12-16)
              and nose (27-35), and above mouth (48-59)

    Args:
      header: Openface header from csv file
      lmks: Line from csv file over which to calculate subregions
      box: Bounding box within which to constrain subregions, formatted [x1, y1, x2, y2]

    Returns:
      list: The subregion, formatted [x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
    """
    def pts2coords(pts: list) -> dict:
        """Converts list of points to dict of {'x': [xVals], 'y': [yVals]}"""
        return {dim: [int(lmks[header.index(f' {dim}_{n}')]) for n in pts] for dim in ['x', 'y']}
    brows = pts2coords(range(17, 27))
    eyes = pts2coords(range(37, 48))
    lEdge = pts2coords(range(0, 5))
    rEdge = pts2coords(range(12, 17))
    nose = pts2coords(range(27, 36))
    mouth = pts2coords(range(48, 60))

    # Origin is in upper left
    forehead = [min(brows['x']), box[1], max(brows['x']), max(brows['y'])]
    lCheek = [max(lEdge['x']), max(eyes['y']), min(nose['x']), min(mouth['y'])]
    rCheek = [max(nose['x']), max(eyes['y']), min(rEdge['x']), min(mouth['y'])]
    return forehead + lCheek + rCheek

def loadBoxes(fname: str, subregions: bool = False) -> list:
    """Load bounding boxes into memory and perform basic format conversions

    This script supports loading openface and annotator formats.
    Spec for openface output is on its wiki: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
    annotator output is csv with the followin fields and types:
      str(label),float(time),int(x1),int(y1),int(x2),int(y2)

    Args:
      fname: The filename for the bounding box source to convert
      subregions: If using openface, calculate subregions (forehead and cheeks).
                  This changes the output format of each box to:
                  [time, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
                  where coords 1-2 are the full bounding box, 3-4 are the forehead,
                  5-6 are the image left cheek, and 7-8 are the image right cheek.

    Returns:
      list: The bounding boxes, formatted [[time, x1, y1, x2, y2], ...]
    """
    import csv
    with open(fname) as f:
        boxes = list(csv.reader(f))
    if boxes[0][0] == 'frame': # openface
        header = [h.strip() for h in boxes[0]]
        boxes = boxes[1:]
        boxes = [[float(x) for x in b] for b in boxes if int(b[header.index('success')]) == 1]
        # range of 35 to include nose in case we have a side view? Doesn't hurt...
        lmkPositions = [[header.index(f'{dim}_{n}') for dim in ['x', 'y']] for n in range(35)]
        #lmkPositions is [[x1, y1], [x2, y2], ...]
        def toBBox(lmks):
            bounds = bboxUtils.landmarks2bbox([[lmks[p] for p in dims] for dims in lmkPositions])
            if subregions:
                bounds += calcSubregions(header, lmks, bounds)
            return bounds
        boxes = [[float(b[header.index('timestamp')])] + toBBox(b) for b in boxes]
    else: # annotator
        # Each box is [str(label), float(time), int(x1), int(y1), int(x2), int(y2)]
        def toBBox(pts):
            pts = [int(p) for p in pts]
            x1, x2 = sorted([pts[0], pts[2]])
            y1, y2 = sorted([pts[1], pts[3]])
            return [x1, y1, x2, y2]
        boxes = [[float(b[1])] + toBBox(b[2:]) for b in boxes]
    return boxes

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Format bounding boxes from an openface or annotator csv file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Path to input csv file')
    parser.add_argument('output', help='Path to output npz file')
    parser.add_argument('--scale', help='Scale factor: original X scale = landmarked video', default=1.0, type=float)
    parser.add_argument('--subregions', help='Calculate subregions in addition to the full bounding box (openface only)', action='store_true')
    parser.add_argument('--split-subregions', help='Split the subregions into multiple files; prepends file names with "srX-"', action='store_true')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    boxes = loadBoxes(args.input, args.subregions)
    boxes = bboxUtils.procBoxes(boxes, scale=args.scale)

    from rPPG.utils import npio
    
    if args.subregions and args.split_subregions:
        from pathlib import Path
        output = Path(args.output)
        for srOffset in range(1, len(boxes[0]), 4):
            npio.save([[b[0]] + b[srOffset:srOffset+4] for b in boxes], None, output.parent / (f'sr{(srOffset-1)//4}-' + output.name))
    else:
        npio.save(boxes, None, args.output)
