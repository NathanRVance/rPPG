#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from pathlib import Path
import json

def train(config: dict, videos: str, output: str, gt: str = None, saveStem: str = 'rpnet', initModel: str = None, saveOptimizerState: bool = False, masks: str = None):
    """Train an rPPG model

    Arguments:
      config: The model and training configuration (must contain splits)
      videos: Path to the directory containing cropped video files
      output: Path to the directory to save the training output
      gt: Path to the directory containing gt (optional)
      saveStem: Name of output models
      initModel: Path to model to initialize training
      saveOptimizerState: Boolean indication whether to save optimizer state
      masks: Path to masks
    """
    import torch
    from progress.bar import IncrementalBar
    from rPPG.utils.dataloader import Dataset
    from rPPG.utils import evaluate
    from rPPG.utils import tables
    from rPPG.utils import modelLoader
    from rPPG.train import test

    outpath = Path(output)
    initEpoch = 0

    # Load model architecture
    if not initModel and outpath.is_dir():
        models = [p for p in outpath.glob(f'{saveStem}_*') if p.is_file()]
        if len(models) >= 1:
            initEpoch = max(int(p.stem.split('_')[1][1:]) for p in models)
            initModel = outpath / f'{saveStem}_e{initEpoch}'
            initEpoch += 1
    model, config = modelLoader.load(modelPath=initModel, config=config)

    # Load datasets (different one for train/val/test)
    DSes = {}
    for name in config['splits']:
        from copy import deepcopy
        cfg = deepcopy(config)
        if name != 'train':
            cfg['training']['augmentation'] = ''
            cfg['model']['fpc'] = cfg.val_test_fpc()
        DSes[name] = Dataset(cfg, config['splits'][name], videos, gtDir=gt, maskDir=masks, ignoreMeta=True)
        if not DSes[name].HRs:
            print(f'ERROR: Must have ground truth gt for training, but none found!')
            exit(1)

    # Ensure output directory exists and save splits
    outpath.mkdir(parents=True, exist_ok=True)
    with (outpath / 'splits.json').open('w') as f:
        json.dump(config['splits'], f)
    # Save config
    with (outpath / 'config.json').open('w') as f:
        json.dump(config, f)

    valRes = {}
    for resName in ['results_bySubject.json', 'results_avg.json']:
        if (outpath / resName).is_file():
            with (outpath / resName).open() as f:
                valRes[resName] = json.load(f)
        else:
            valRes[resName] = []

    # Now start training!
    for epoch in range(initEpoch, config.num_epochs()):
        _, _, _, _, trainLoss, trainLossParts, _ = DSes['train'].infer(model, train=True, barLabel='Training')
        # Validate the model
        predWaves, predHR, gtWaves, gtHR, loss, lossParts, predFFT = DSes['val'].infer(model) 
        bySubject, valAvg = evaluate.evaluate(predWaves, predHR, gtWaves, gtHR, config.fps(), masks=DSes['val'].masks, fft=predFFT, multiprocessing=config.multiprocessing())
        valRes['results_bySubject.json'].append(bySubject)
        valAvg.update(test.formatLossParts(lossParts, loss))
        valAvg.update(test.formatLossParts(trainLossParts, trainLoss, suffix='-Train'))
        tables.printTable(test.filterResults(valAvg, config.columns()))
        valRes['results_avg.json'].append(valAvg)
        # Save model
        savePath = outpath / f'{saveStem}_e{epoch}'
        modelLoader.saveModelData(model.state_dict(), model.optimizer_state_dict if saveOptimizerState else None, config, savePath)
        print(f'Saved model to {savePath}\n')
        # Save validation results (overwrites previous, but ok because we append)
        for fname, dic in valRes.items():
            with (outpath / fname).open('w') as f:
                json.dump(dic, f)

    # Finally, test on best model
    from rPPG.utils import getBestModel
    modelPath = outpath / f'{saveStem}_e{getBestModel.getBestEpoch(valRes["results_avg.json"])}'
    print(f'Testing model {modelPath}')
    testModel, testConfig = modelLoader.load(modelPath, None)
    test.test(testModel, testConfig, videos, gt=gt, masks=masks, save=str(modelPath)+'-test/')

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Train over a dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', help='Path to json-formatted configuration (see readme for details)')
    parser.add_argument('gt', nargs='?', help='Path with ground truth waves where each filename is formatted subID.npz. By default uses gt bundled with the videos.')
    parser.add_argument('videos', help='Path to videos directory where each filename is formatted subID.npz')
    parser.add_argument('output', help='Path to save trained models')
    parser.add_argument('--splits', help='Path to json-formatted splits file (see readme for details); by default does random splits (see --randomSplits)')
    parser.add_argument('--randomSplits', nargs=3, help='Use random splits. Arguments are train, val, test ratios', default=[60, 20, 20], metavar=('TRAIN', 'VAL', 'TEST'), type=float)
    parser.add_argument('--saveStem', metavar='STEM', help='Model name when saving at each epoch; saves to OUTPUT/STEM_eEPOCH. Default is model architecture.')
    parser.add_argument('--initModel', help='Initialize model as the model at this path')
    parser.add_argument('--clean', action='store_true', help='Wipe out data in OUTPUT and start from scratch. Default action is to resume training from these models.')
    parser.add_argument('--saveOptimizerState', help='Save the optimizer state in the model as optimizer_state_dict', action='store_true')
    parser.add_argument('--masks', help='Path to directory with masks as generated by utils/cleanLabels.py --outIntervals. By default uses masks bundled with the videos.')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
   
    from rPPG.utils import config

    # Load the config
    with open(args.config) as f:
        config = config.Config(json.load(f))

    if not args.saveStem:
        args.saveStem = config.architecture()
    
    # Determine splits
    if args.splits:
        with open(args.splits) as f:
            splits = json.load(f)
    elif 'splits' in config:
        splits = config['splits']
    else:
        # Get list of subIDs
        subIDs = [fname.stem for fname in Path(args.videos).iterdir()]
        if args.gt: # Take subset that also are in gt
            subIDs = [fname.stem for fname in Path(args.gt).iterdir() if fname.stem in subIDs]
        from rPPG.utils import genSplits
        splits = genSplits.genSplits(subIDs, {'train': args.randomSplits[0], 'val': args.randomSplits[1], 'test': args.randomSplits[2]})

    # Put splits in config
    config['splits'] = splits

    outpath = Path(args.output)
    if args.clean:
        # Wipe everything out from previous run
        import shutil
        if outpath.is_dir():
            shutil.rmtree(outpath)

    train(config, args.videos, args.output, gt=args.gt, saveStem=args.saveStem, initModel=args.initModel, saveOptimizerState=args.saveOptimizerState, masks=args.masks)
