#!/usr/bin/env python3
from rPPG.utils import modelLoader
from rPPG.utils import npio
from rPPG.utils import augmentations
import torch
import numpy as np
from copy import deepcopy
import cv2
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model activations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', help='Path to model')
    parser.add_argument('sample', help='Path to data sample')
    parser.add_argument('output', help='Path to write output video')
    parser.add_argument('--start', type=float, default=0, help='Start time (s)')
    parser.add_argument('--end', type=float, default=10, help='End time (s)')
    parser.add_argument('--layerTypes', nargs='+', help='Layer types to compare, e.g., "Conv3d", "BatchNorm3d", etc. Default compares all.', default=None)
    parser.add_argument('--cmap', default='gray', help='Colormap to use when plotting')

    args = parser.parse_args()

    model, config = modelLoader.load(args.model, None)
    config['training']['augmentation'] = ''
    data, meta = npio.loadMeta(args.sample)
    data, _ = augmentations.applyAugmentations(data, None, None, [int(meta.fps()*args.start), int(meta.fps()*args.end)], config)
    data = data[np.newaxis, ...] # Simulate batch size = 1
    data = torch.from_numpy(data).float()
    device = next(model.parameters()).device
    data = data.to(device)
    
    layerOuts = []
    for module in reversed(model.forward_stream):
        if args.layerTypes is None or type(module).__name__ in args.layerTypes:
            print(f'Using layer {type(module).__name__}')
            outputs = model.forward_stream(data)
            layerOuts.append(np.squeeze(deepcopy(outputs.detach().cpu().numpy())))
            del outputs
        else:
            print(f'Skipping layer {type(module).__name__}')
        model.forward_stream = model.forward_stream[:-1]
    # Data is currently shape C,T,X,Y
    def fixLayerOut(lo):
        if len(lo.shape) == 4:
            # Might be horrendously busy, but just add all channels together:
            lo = np.sum(lo, axis=0)
        #elif len(lo.shape) == 2:
        #    lo = np.moveaxis(lo, 0, -1) # Convert C,T to T,C
        elif len(lo.shape) == 1:
            # Convert T to T,X,Y
            lo = lo[:, np.newaxis, np.newaxis]
        lo -= lo.min()
        return lo*255.0/lo.max()
    layerOuts = [fixLayerOut(lo) for lo in reversed(layerOuts)]

    # Generate plots
    data = np.squeeze(data.cpu().numpy())
    frames = []
    out = None
    for i in IncrementalBar('Plotting results', max=len(data), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(range(len(layerOuts[0]))):
        grid = []
        for layerOut in [l[i] for l in layerOuts]:
            fig = plt.figure()
            plt.imshow(layerOut, vmin=0, vmax=255, cmap=args.cmap)
            plt.tight_layout()
            fig.canvas.draw()
            img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            plt.close('all')
            grid.append(img)
        cols = int(np.ceil(np.sqrt(len(grid))))
        rows = int(np.ceil(len(grid) / cols))
        h, w, c = grid[0].shape
        frame = np.zeros((rows*h, cols*w, c))
        for i, img in enumerate(grid):
            hOff = (i % cols) * w
            vOff = (i // cols) * h
            frame[vOff:vOff+h, hOff:hOff+w] = img
        frame = np.uint8(frame)
        # Scale width down to 1080
        frame = cv2.resize(frame, (1080, int(frame.shape[0]*1080/frame.shape[1])), interpolation=cv2.INTER_AREA)
        if out is None:
            out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), meta.fps(), (frame.shape[1], frame.shape[0]))
        out.write(frame)
