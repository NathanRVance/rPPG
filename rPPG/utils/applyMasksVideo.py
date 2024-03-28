#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Apply a mask to a video file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mask', help='npz formatted mask to apply')
    parser.add_argument('video', help='video file to mask')
    parser.add_argument('--invertMask', action='store_true', help='Invert the mask to remove everything not in the interval')
    parser.add_argument('output', help='output video file')
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from rPPG.utils import npio
    mask, maskmeta = npio.loadMeta(args.mask)
    video, vidmeta = npio.loadMeta(args.video)
    gt = vidmeta.gt()

    if args.invertMask:
        frame = 0
        inverted = []
        for start, end in mask:
            if frame < start:
                inverted.append([frame, start])
            frame = end+1
        end = 0 if len(mask) == 0 else mask[-1][1]
        if min(len(video), len(gt)) > end:
            inverted.append([end, min(len(video), len(gt))])
        mask = inverted

    print(f'Applying mask: {mask}')
    from rPPG.utils import masks
    video = masks.applyMaskSkip(video, mask)
    gt = masks.applyMaskSkip(gt, mask)
    vidmeta.setGt(gt)

    npio.save(video, vidmeta, args.output)
    
