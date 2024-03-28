#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse, argcomplete

parser = argparse.ArgumentParser(description='Demonstrate the rPPG pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('datadir', help='Directory containing either the video/signal on which to operate, or a directory for each subject with a video/signal')
parser.add_argument('--results', help='Directory to place results. By default places them under datadir.')
parser.add_argument('--video', help='Name of the video file(s) to use', default='compressed_RGBVideo.avi')
parser.add_argument('--gt', help='Name of ground truth signal file(s) to use. Globs are accepted, and will be expanded such that routines in combineWaves.py are used.', default='oximeterData.csv')
parser.add_argument('--skipGt', action='store_true', help='Skip ground truth processing')
parser.add_argument('--gtCols', default=('hr', 'o2', 'signal'), nargs='+', help='Data column headers in gt csv files')
parser.add_argument('--maskDelta', type=float, default=7, help='If nonzero, calculate masks for ground truth with maximum delta of HR in units of bpm/second used to identify error sections of the waveform')
parser.add_argument('--landmarker', help='Landmarker to use', choices=['openface', 'mediapipe'], default='mediapipe')
parser.add_argument('--average', nargs='+', type=int, help='Frames to "average" when traversing the video and boxes, used to obtain lower FPS output. When average=0, iterates over the video normally. Multiple parameters result in "aX-" prepended to the output file name, where X is the average value.', default=[0])
parser.add_argument('--shiftBaselines', nargs='+', default=[], choices=['CHROM', 'POS'], help='Baseline technique(s) to use for adjusting signals. Default skips this.')
parser.add_argument('--maxShift', type=float, default=1, help='Maximum amonut to shift signal, in seconds')
parser.add_argument('--overwrite', help='Recalculate and overwrite existing results', action='store_true')
parser.add_argument('--model', help='Test rPPG model on data')
parser.add_argument('--kfolds', type=int, help='Perform kfolds training once the data is preprocessed')

argcomplete.autocomplete(parser)
args = parser.parse_args()

from pathlib import Path
import glob
from utils import npio
import subprocess
from utils.metadata import Metadata

paths = {}

datadir = Path(args.datadir)
if (datadir / args.video).is_file():
    paths = {Path(args.video).stem: {'video': datadir / args.video, 'gt': datadir / args.gt}}
else:
    paths = {p.name: {'video': p / args.video, 'gt': p / args.gt} for p in datadir.iterdir() if p.is_dir()}
if args.skipGt:
    for p in paths:
        del paths[p]['gt']

for ps in paths.values():
    for p in ps.values():
        if not p.is_file() and not glob.glob(str(p)):
            print(f'WARN: {p} is not a file!')
            #exit(1)

def procSub(subID, video, signal, args):
    def getResFname(stage, suffix):
        p = Path(args.results) / stage / f'{subID}.{suffix}' if args.results else Path(video).parent / 'results' / f'{stage}.{suffix}'
        p.parent.mkdir(parents = True, exist_ok = True)
        return p
    if not args.skipGt and (not Path(signal).is_file() or Path(signal).stat().st_size == 0) and len(glob.glob(str(signal))) < 2:
        print(f'ERROR: Signal {signal} has size 0; skipping...')
        return
    if not Path(video).is_file():
        print(f'ERROR: Could not find video {video}!')
        return
    meta = Metadata.fromVideo(video)
    bboxesRes = getResFname('bboxes', 'npz')
    if args.overwrite or not bboxesRes.is_file():
        from utils import bboxUtils
        if args.landmarker == 'openface':
            landmarks = getResFname('landmarks', 'csv')
            if args.overwrite or not landmarks.is_file():
                subprocess.run(['FeatureExtraction', '-f', video, '-out_dir', landmarks.parent, '-of', landmarks.stem, '-2Dfp'])
            from bboxes import formatBBoxes
            boxes = bboxUtils.procBoxes(formatBBoxes.loadBoxes(landmarks))
        else:
            from bboxes import bboxesMp
            boxes = bboxesMp.calcMP(str(video))
        npio.save(boxes, meta, getResFname('bboxes', 'npz'))
    if not args.skipGt:
        gtRes = getResFname('gt', 'npz')
        if args.overwrite or not gtRes.is_file():
            from utils import cvtData
            gtDatas = []
            for signalFile in glob.glob(str(signal)):
                gtData, gtMeta = cvtData.cvtData(signalFile, outputFPS=meta.fps())
                gtMeta.setLocation(Path(signalFile).stem.split('_')[-1])
                #if gtMeta.location() != 'cms50ea':
                #    gtData = gtMeta['channels']['1']
                for i, key in enumerate(args.gtCols):
                    gtMeta[key] = gtMeta['channels'][str(i)]
                if gtMeta.gt() is not None:
                    gtData = gtMeta.gt()
                gtMeta.setData(gtData)
                gtDatas.append(gtMeta)
            if len(gtDatas) == 1:
                gtData = gtDatas[0]
            else:
                from utils import combineWaves
                gtData = combineWaves.combineWaves(gtDatas)
            npio.save(gtData.data(), gtData, gtRes)
            # We will embed after calculating baselines, but do it once now to catch hr, o2, bp, etc.
            for key in args.gtCols:
                if key in meta:
                    print(f'WARN: "{key}" already a key in metadata')
                meta[key] = gtData[key]
        if args.shiftBaselines:
            signalsRes = getResFname('signals', 'npz')
            if args.overwrite or not signalsRes.is_file():
                print('Calculating baselines')
                from baselines import baselinesPreprocess
                signals = baselinesPreprocess.calcSignals(video, bboxes=npio.load(bboxesRes))
                npio.save(signals, meta, signalsRes)
            from baselines import calcBaselines
            CHROM, POS, m = calcBaselines.loadAndCalculate(signals=signalsRes, video=video)
            bl = {'CHROM': CHROM, 'POS': POS}
            from baselines import cvtSignal
            l = cvtSignal.shiftSignal(npio.load(gtRes), [bl[b] for b in args.shiftBaselines], int(args.maxShift*meta.fps()), meta.fps())
            npio.save(l, m, gtRes)
        meta.embedMetadata('signal', gtRes)
        if args.maskDelta:
            maskRes = getResFname('masks', 'npz')
            if args.overwrite or not maskRes.is_file():
                from utils import masks
                mask, _ = masks.calcMask(npio.load(gtRes), meta.fps(), maxDelta=args.maskDelta)
                npio.save(mask, meta, maskRes)
            meta.embedMetadata('mask', maskRes)
    croppedRes = getResFname('cropped', 'hdf5')
    if args.overwrite or not croppedRes.is_file():
        from crop import crop
        crop.crop(video, [npio.load(bboxesRes)], meta, croppedRes, average=args.average)

for subID, ps in paths.items():
    print(f'Processing subID: {subID}')
    procSub(subID, ps['video'], None if args.skipGt else ps['gt'], args)

if args.results and args.model:
    from train import test
    from utils import modelLoader
    model, config = modelLoader.load(args.model)
    test.test(model, config, Path(args.results) / 'cropped', testAll=True, save=Path(args.results) / 'test')

if args.results and args.kfolds:
    from utils import genSplits
    from train import train
    import json
    folds = genSplits.genKFolds([fname.stem for fname in (Path(args.results) / 'cropped').iterdir()], args.kfolds)
    outdir = Path(args.results) / 'splits'
    outdir.mkdir(exist_ok=True, parents=True)
    for k, fold in enumerate(folds):
        with (outdir / f'{k}-fold.json').open('w') as f:
            json.dump(fold, f)
    # get a metadata specimen
    meta = Metadata.fromVideo(list(paths.values())[0]['video'])
    for k, fold in enumerate(folds):
        print(f'Training fold {k}')
        config = {"model": {"architecture": "CNN3D", "tk": 5, "channels": "rgb", "frame_width": 64, "frame_height": 64, "fpc": 136, "step": 68, "fps": meta.fps()},
                "training": {"num_workers": 4, "num_epochs": 40, "augmentation": "fig", "dropout": 0.5, "batch_size": 4, "lr": 0.0001, "masks": True},
                "evaluation": { "hr_method": "fft", "hz_low": 0.66666, "hz_high": 3.0, "fft_window": 10.0, "move_cost": 10.0, "smooth_method": "none", "delta_limit": 6.0, "smooth": 5.0},
                "splits": fold}
        train.train(config, Path(args.results)/'cropped', Path(args.results)/'training'/fold)
