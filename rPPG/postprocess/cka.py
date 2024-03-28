#!/usr/bin/env python3
from torch_cka import CKA
from rPPG.utils import modelLoader
from rPPG.utils.dataloader import Dataset
from rPPG.utils import tables
from pathlib import Path
import torch

def cka(model1, model2, names, layerTypes, videos, gt=None, batchsize=4, separateDataloaders=False, videos2=None, gt2=None, shuffle=False):
    models = []
    layersToCompare = []
    dataloaders = []
    for i, modelPath in enumerate([model1, model2]):
        model, config = modelLoader.load(modelPath, None, None)
        models.append(model)
        layers = []
        #for module in model.forward_stream):
        for layerName, module in model.named_modules():
            if layerTypes is None or type(module).__name__ in layerTypes:
                layers.append(layerName)
        layersToCompare.append(layers)
        if not separateDataloaders and not videos2 and len(dataloaders) > 0:
            break
        config['training']['augmentation'] = ''
        config['training']['masks'] = False
        config['training']['batch_size'] = batchsize
        vidDir = videos
        gtDir = gt
        if i == 1 and videos2 is not None:
            vidDir = videos2
            gtDir = gt2
        dataset = Dataset(config, [Path(vid).stem for vid in Path(vidDir).iterdir()], vidDir, gtDir=gtDir, ignoreMeta=True)
        dataset.root = None
        dataset.lenOverride = len(dataset) - len(dataset) % batchsize
        print('Summary of test data:')
        tables.printTable(tables.mergeTables(dataset.getStats()))
        dataloaders.append(torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=batchsize, drop_last=True))
    
    cka = CKA(*models, model1_name=names[0], model2_name=names[1], model1_layers=layersToCompare[0], model2_layers=layersToCompare[1], device='cuda')
    cka.compare(*dataloaders)
    #try:
    #    cka.compare(*dataloaders)
    #except AssertionError as e:
    #    print(f'WARN: {e}')
    # Free models and dataloader
    for model in models:
        del model
    del dataloaders
    return cka

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate the Centered Kernel Alignment (CKA) for two or more networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('gt', nargs='?', help='Path with ground truth waves where each filename is formatted subID.npz. By default uses gt bundled with the videos.')
    parser.add_argument('videos', help='Path to videos directory where each filename is formatted subID.npz')
    parser.add_argument('model1', help='Network to compare')
    parser.add_argument('model2', help='Network to compare')
    parser.add_argument('--names', nargs=2, help='Names for models', default=['model1', 'model2'])
    parser.add_argument('--layerTypes', nargs='+', help='Layer types to compare, e.g., "Conv3d", "BatchNorm3d", etc. Default compares all.', default=None)
    parser.add_argument('--separateDataloaders', action='store_true', help='Force using separate dataloaders (i.e., for different fpc). Otherwise creates a dataloader based on config in model1.')
    parser.add_argument('--ds2', nargs='+', help='Use proveded dataset for model2. Arguments are gt (optional), videos.')
    parser.add_argument('--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('--shuffleDS', action='store_true', help='Shuffle the dataset. Default does not shuffle.')
    parser.add_argument('--save', help='Path to save .npy results')
    parser.add_argument('--plot', help='Path to plot results')

    args = parser.parse_args()

    videos2 = None
    gt2 = None
    if args.ds2:
        if len(args.ds2) == 1:
            videos2 = args.ds2[0]
        else:
            videos2 = args.ds2[1]
            gt2 = args.ds2[0]

    ckaRes = cka(args.model1, args.model2, args.names, args.layerTypes, args.videos, args.gt, args.batchsize, args.separateDataloaders, videos2, gt2, shuffle=args.shuffleDS)

    if args.plot:
        ckaRes.plot_results(save_path=args.plot)

    if args.save:
        results = ckaRes.export()
        import numpy as np
        np.save(args.save, results)
