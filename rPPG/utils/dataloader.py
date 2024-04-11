#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import numpy as np
import torch
from progress.bar import IncrementalBar
from multiprocessing import Pool
import cv2
from rPPG.utils import hr
from rPPG.utils import npio
from rPPG.utils import losses
from rPPG.utils import modelExecutor
from rPPG.utils import masks as msk
from rPPG.utils import augmentations

class Dataset(VisionDataset):
    """
    This class is the Dataset for pytorch.

    The pytorch framework treats it as a list, calling __len__() and __getitem__()
    to read elements from the dataset. Each value returned by __getitem__() then
    is used for one evaluation of the model.

    We also include an infer() method to evaluate or train on the dataset.
    """

    def __init__(self, config: dict, subIDs: list, videoDir: str, gtDir: str = None, maskDir: str = None, ignoreMeta: bool = False):
        """The initialization performs the following steps:
        1. Load the subjects contained in subIDs dataset into memory
        2. Convert video channels to the order expected by the model
        3. Assert that the framerate of the videos match the model
        4. Load masks (if provided)
        5. Load gt (if provided)
        6. Precompute heartrate (if gt signals are provided)

        Args:
          config: The configuration for the model as returned by modelLoader.load, or passed into train.py
          subIDs: The list of subjects to be loaded. This is a subset of the filename stems in videoDir (and gtDir)
          videoDir: The directory containing video inputs, with each file as subID.npz, or a specific video file.
          gtDir: The directory containing ground truth waves, with each file as subID.npz (optional)
          maskDir: The directory containing masked intervals for ground truth waves, or a specific mask file (optional)
          ignoreMeta: If True, do not throw an error for having mismatched fps in the metadata
        """
        self.lenOverride = 0
        self.config = config
        # Load the videos: {subID: video}
        if Path(videoDir).is_dir():
            fnames = [fname for fname in Path(videoDir).iterdir() if fname.stem in subIDs]
        else: # Is actually a file
            fnames = [Path(videoDir)]
        self.metadata = {}
        self.videos = {}
        self.waves = None
        self.masks = None
        for fname in IncrementalBar('Loading videos', max=len(fnames), suffix='%(index)d/%(max)d - %(elapsed)d s').iter(fnames):
            video, metadata = npio.loadMeta(fname)
            self.videos[fname.stem] = video
            self.metadata[fname.stem] = metadata
        for subID, meta in self.metadata.items():
            if not meta:
                continue
            if not gtDir and meta.gt() is not None:
                if self.waves is None:
                    self.waves = {}
                self.waves[subID] = np.array(meta.gt())
            if not maskDir and meta.mask() and self.config.masks():
                if self.masks is None:
                    self.masks = {}
                self.masks[subID] = meta.mask()
            if not ignoreMeta:
                # Ensure FPS matches
                vidFPS = meta.fps()
                modelFPS = self.config.fps()
                if abs(modelFPS - vidFPS) / vidFPS > 0.25:
                    print(f'ERROR: Using a model trained at {modelFPS} fps to infer over a video at {vidFPS} fps')
                    exit(1)
        for subID in subIDs:
            if subID not in self.videos:
                print(f'ERROR: Could not find {subID}.npz in {videoDir}')
        if maskDir:
            if Path(maskDir).is_dir():
                self.masks = {fname.stem: npio.load(fname) for fname in Path(maskDir).iterdir() if fname.stem in subIDs}
            else: # Is actually a file
                mask = npio.load(maskDir)
                self.masks = {subID: mask for subID in subIDs}
        if gtDir:
            # Load the gt (waves): {subID: wave}
            self.waves = {fname.stem: npio.load(fname) for fname in Path(gtDir).iterdir() if fname.stem in subIDs}
            # Normalize wave
            for subID, wave in self.waves.items():
                self.waves[subID] = (wave - np.mean(wave)) / np.std(wave)
        if self.waves:
            # Chop waves and videos down to the same length
            for subID in list(self.waves.keys()):
                if subID not in self.videos:
                    continue
                minLen = min(len(dic[subID]) for dic in [self.videos, self.waves])
                # No need to shorten videos because waves are preferred in chunkVideos()
                self.waves[subID] = self.waves[subID][:minLen]
        
        self.chunks = self.chunkVideos()

        # Make all dicts the same subset
        dicts = [self.chunks, self.videos]
        if self.masks:
            dicts.append(self.masks)
        if self.waves:
            dicts.append(self.waves)
        keys = set(dicts[0].keys())
        for dic in dicts[1:]:
            keys = keys.intersection(set(dic.keys()))
        for dic in dicts:
            for key in list(dic.keys()):
                if key not in keys:
                    del dic[key]
        
        self.HRs = None
        if self.waves:
            # Chop down waves to chunk size (avoids warnings in evaluation)
            for subID in self.waves:
                self.waves[subID] = self.waves[subID][:self.chunks[subID][-1][1]]
            # Precompute HRs
            self.HRs = {subID: HR for subID, (HR, _) in doCalcHR(self.waves, self.config).items()}
            if self.masks:
                self.waves, self.HRs = msk.applyMaskDicts([self.waves, self.HRs], self.masks, method='linear')

    def getStats(self, stddevWindow=0):
        def getStatsSingle(subID):
            stats = {'numframes': len(self.waves[subID]), 'time': len(self.waves[subID]) / self.metadata[subID].fps(), 'avgHR': float(np.mean(self.HRs[subID])), 'avgHRstddev': float(np.std(self.HRs[subID]))}
            if stddevWindow:
                w = int(stddevWindow * self.metadata[subID].fps())
                stats[f'stddevWindow={stddevWindow}'] = float(np.mean([np.std(self.HRs[subID][i:i+w]) for i in range(len(self.HRs[subID]-w))]))
            return stats
        return [getStatsSingle(subID) for subID in self.waves.keys()]

    def calcHistograms(self):
        hists = [[] for i in range(3)]
        print(f'Applying {self.config.normalization()} normalization')
        for subID in self.chunks:
            for start, end in self.chunks[subID]:
                clip = self.videos[subID][start:end]
                if self.config.normalization() != 'scale':
                    clip = augmentations.normalize(clip, self.config.normalization()).astype(np.uint8)
                for i in range(3):
                    hists[i].append(cv2.calcHist(clip, [i], None, [256], [0,256]))
        hists = [sum(h) for h in hists]
        return hists, 'bgr'

    def chunkVideos(self):
        """Calculate start/end frames for each video given fpc, step, and potentially masks.
        
        Returns:
          dict: {subID: [(start, end), ...]}
        """
        ret = {}
        # Prefer waves if available because of loading time
        media = self.waves if self.waves else self.videos
        for subID, value in media.items():
            fpc = self.config.fpc()
            index = 0
            nextMask = None
            maskIndex = 0
            if self.masks and subID in self.masks and len(self.masks[subID]) > 0:
                nextMask = self.masks[subID][0]
            if fpc == 0:
                fpc = len(value)
                if nextMask is not None: # Uh oh
                    print('WARN: fpc was set to 0 (i.e., full length) and masks were provided. This may cause errors down the line.')
            step = int(fpc//2)
            while True:
                if nextMask is not None and max(index, nextMask[0]) < min(index+fpc, nextMask[1]):
                    index = nextMask[1]
                    if len(self.masks[subID]) > maskIndex+1:
                        maskIndex += 1
                        nextMask = self.masks[subID][maskIndex]
                if index+fpc > len(value):
                    break
                if subID not in ret:
                    ret[subID] = []
                ret[subID].append((index, index+fpc))
                index += step
        return ret

    def __len__(self):
        if self.lenOverride:
            return self.lenOverride
        return sum(len(chunk) for chunk in self.chunks.values()) * self.config.clip_batch_duplicate()

    def __getitem__(self, index) -> (np.ndarray, np.ndarray, str):
        """In addition to selecting the appropriate {fpc}-length clip from the
        appropriate video, performs any necessary formatting and augmentations.

        Returns:
          np.ndarray: Pointer to the clip on the GPU
          np.ndarray: Pointer to the wave on the GPU (empty if no gt provided)
          str: Subject ID
        """
        # Determine which subID
        for subID in sorted(self.chunks.keys()):
            chunks = self.chunks[subID]
            thisLen = len(chunks) * self.config.clip_batch_duplicate()
            if thisLen <= index:
                index -= thisLen
                continue
            # This is the one
            start, end = chunks[index//self.config.clip_batch_duplicate()]
            configToPass = self.config
            if index % self.config.clip_batch_duplicate() != 0 and 's' not in configToPass.augmentation():
                from rPPG.utils.config import Config
                configToPass = Config(configToPass)
                configToPass['training']['augmentation'] += 's'
            clip, extras = augmentations.applyAugmentations(
                    self.videos[subID],
                    None if not self.waves else self.waves[subID],
                    None if not self.HRs else self.HRs[subID],
                    (start, end),
                    configToPass)
            clip = torch.from_numpy(clip).float()
            if 'wave' in extras:
                extras['wave'] = torch.from_numpy(extras['wave']).float()
            extras['subID'] = subID
            #print(f'Returning from subID {subID}, shape {clip.shape}')
            return clip, extras

    def infer(self, model, train: bool = False, barLabel: str = 'Inferring', skipPostprocess: bool = False) -> (dict, dict, dict, dict, float, dict):
        """ Performs inferrence over the dataset
        
        Args:
          model: The model on which inferrence will be performed
          train: Set to True to train the model using backpropogation
          barLabel: Printed label on the progress bar
          skipPostprocess: Returns empty arrays for heartrate and fft

        Returns:
          dict: Predicted waves as {subID: np.ndarray}
          dict: Predicted heartrate as {subID: np.ndarray}
          dict: Ground truth waves as {subID: np.ndarray} or None if gt not provided
          dict: Ground truth heartrate as {subID: np.ndarray} or None if gt not provided
          float: Loss, if gt provided
          dict: Predicted FFT as {subID: np.ndarray}
        """
        from copy import deepcopy
        # Initialize the data loader
        shuffle=train and self.config.shuffle()
        loader = torch.utils.data.DataLoader(self, batch_size=self.config.batch_size(), shuffle=shuffle, num_workers=self.config.num_workers())
        #loader = torch.utils.data.DataLoader(self, batch_size=1, shuffle=False, num_workers=1) # test/val loader
        optimizer = None
        if train:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr())
            if model.optimizer_state_dict is not None:
                optimizer.load_state_dict(model.optimizer_state_dict)
        predictions, loss, lossParts = modelExecutor.execModel(model, loader, optimizer=optimizer, train=train, barLabel=barLabel)
        # Postprocess
        predWavesOnly = {} # Differs from predictions if out_channels > 1
        predHR = {}
        predFFT = {}
        if not train and not skipPostprocess:
            for subID in predictions:
                waveLen = self.chunks[subID][-1][1]
                wave = np.squeeze(np.zeros((waveLen, self.config.out_channels())))
                for i, (window, (start, end)) in enumerate(zip(predictions[subID], self.chunks[subID])):
                    hw = np.hanning(end-start)
                    if i == 0:
                        hw[:len(hw)//2] = hw[len(hw)//2]
                    if i == len(self.chunks[subID])-1:
                        hw[len(hw)//2:] = hw[len(hw)//2]
                    if self.config.out_channels() > 1:
                        hw = hw[:,np.newaxis]
                    #print(f'subID: {subID}, i: {i}, window: {window.shape}, start: {start}, end: {end}, wave: {wave.shape}')
                    wave[start:end] = wave[start:end] + hw * np.squeeze(window)
                predictions[subID] = (wave - np.mean(wave)) / np.std(wave)
                if self.config.out_channels() > 1:
                    predWavesOnly[subID] = predictions[subID][:,0]
                else:
                    predWavesOnly[subID] = predictions[subID]
            for subID, (HR, fft) in doCalcHR(predWavesOnly, self.config).items():
                predHR[subID] = HR
                predFFT[subID] = fft
            if self.masks:
                predictions, predHR = msk.applyMaskDicts([predictions, predHR], self.masks, method='linear')
        elif train:
            model.optimizer_state_dict = optimizer.state_dict()
        return predictions, predHR, self.waves, self.HRs, loss, lossParts, predFFT

def doCalcHR(waves, config):
    kwargs = {'hrCalculator': config.hr_method(), 'lowHz': config.hz_low(), 'highHz': config.hz_high(), 'window': config.fft_window(), 'moveCost': config.move_cost(), 'deltaLimit': config.delta_limit(), 'smooth': config.smooth(), 'skipFFT': config.skip_fft()}
    sm = config.smooth_method()
    if sm == 'none':
        kwargs['deltaLimit'] = 0
        kwargs['smooth'] = 0
    elif sm == 'delta_limit':
        kwargs['smooth'] = 0
    elif sm == 'smooth':
        kwargs['deltaLimit'] = 0
    return hr.calcHR(waves, config.fps(), **kwargs)

if __name__ == '__main__':
    import argparse, argcomplete
    
    parser = argparse.ArgumentParser(description='Load a dataset and calculate basic metrics', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', help='Path to json-formatted configuration (see readme for details)')
    parser.add_argument('gt', nargs='?', help='Path with ground truth waves where each filename is formatted subID.npz. By default uses gt bundled with the videos.')
    parser.add_argument('videos', help='Path to videos directory where each filename is formatted subID.npz')
    parser.add_argument('--masks', help='Path to directory with masks as generated by utils/cleanLabels.py --outIntervals. By default uses masks bundled with the videos.')
    parser.add_argument('--saveStats', help='Path to save dataset statistics')
    parser.add_argument('--stddevWindow', default=0, type=float, help='Sliding window for calculating HR stddev, in seconds')
    parser.add_argument('--saveHistogram', help='Path to save the color histogram')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Load the config
    import json
    from rPPG.utils import config
    with open(args.config) as f:
        config = config.Config(json.load(f))

    subIDs = [fname.stem for fname in Path(args.videos).iterdir()]
    if args.gt: # Take subset that also are in gt
        subIDs = [fname.stem for fname in Path(args.gt).iterdir() if fname.stem in subIDs]

    ds = Dataset(config, subIDs, args.videos, gtDir=args.gt, maskDir=args.masks, ignoreMeta=True)
    from rPPG.utils import tables
    print('Summary of Dataset:')
    stats = ds.getStats(args.stddevWindow)
    tables.printTable(tables.mergeTables(stats))
    if args.saveStats:
        with Path(args.saveStats).open('w') as f:
            json.dump(stats, f)
    
    if args.saveHistogram:
        # Do histograms
        hists, channels = ds.calcHistograms()
        from matplotlib import pyplot as plt
        for hist, channel in zip(hists, channels):
            plt.plot(hist, color=channel, label=f'{channel} channel')
        plt.legend()
        plt.ylabel('Frequency')
        plt.yticks([])
        plt.xlabel('Channel value')
        plt.tight_layout()
        plt.show()
        plt.savefig(args.saveHistogram)
