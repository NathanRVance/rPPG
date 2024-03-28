#!/usr/bin/env python3
import numpy as np
import copy
from rPPG.utils import cvtData
import json
from pathlib import Path

def mergeDict(dict1, dict2):
    """dict1 has precidence"""
    if not dict2:
        return dict1
    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val
        elif type(val) == dict:
            dict1[key] = mergeDict(dict1[key], dict2[key])
    return dict1

class Metadata(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)
        '''
        try:
            dict.__init__(self, *args)
        except:
            dict.__init__(self)
        '''

    def fps(self):
        return eval(str(self['streams'][0]['avg_frame_rate']))

    def setFPS(self, fps: float):
        oldFPS = 0
        try:
            oldFPS = self.fps()
        except:
            pass
        if 'streams' not in self:
            self['streams'] = [{}]
        for stream in self['streams']:
            stream['r_frame_rate'] = fps
            stream['avg_frame_rate'] = fps
            if oldFPS and 'nb_frames' in stream:
                stream['nb_frames'] = eval(str(stream['nb_frames'])) * (fps / oldFPS)
            if oldFPS and 'time_base' in stream:
                stream['time_base'] = eval(str(stream['time_base'])) / (fps / oldFPS)

    def setStartTime(self, starttime: float):
        self['starttime'] = float(starttime)

    def startTime(self) -> float:
        return self['starttime']

    def setLocation(self, location: str):
        self['location'] = location

    def location(self) -> str:
        return self['location']

    def setData(self, data):
        self['data'] = data

    def data(self):
        return self['data']

    def gt(self):
        if 'gt' in self:
            return self['gt']
        elif 'signal' in self:
            return self['signal']
        elif 'label' in self:
            return self['label']
        return None

    def label(self):
        return self.gt()

    def setGt(self, gt: list):
        self['gt'] = gt

    def mask(self):
        if 'mask' in self:
            return self['mask']
        return None

    def setMask(self, mask):
        self['mask'] = mask

    def makeSkipped(self, skip):
        meta2 = Metadata(copy.deepcopy(self))
        # Adjust metadata for skip
        meta2.setFPS(self.fps() / (skip+1))
        for key in ['label', 'gt', 'signal', 'mask']:
            if key in meta2:
                meta2[key] = cvtData.cvtFPS(meta2[key], 1, 1/(skip+1))
        return meta2

    def getPath(self, keyPath: str):
        ret = self
        for p in keyPath.split('/'):
            ret = ret[p]
        return ret

    def embedMetadata(self, keyPath: str, fname: str):
        metaFile = Path(fname)
        if metaFile.suffix == '.json':
            with metaFile.open() as f:
                newMeta = json.load(f)
        else:
            from rPPG.utils import npio
            newMeta = npio.load(metaFile)
            try:
                newMeta = newMeta.tolist()
            except:
                pass
        for p in reversed([p for p in keyPath.split('/') if p]):
            newMeta = {p: newMeta}
        self.update(mergeDict(newMeta, self))

    @staticmethod
    def fromVideo(fname):
        import subprocess
        return Metadata(json.loads(subprocess.check_output(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', fname])))

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Manipulate metadata on a model or data item', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('object', help='Object to which to add/query metadata')
    parser.add_argument('--metadata', help='Metadata to add to the model')
    parser.add_argument('--metadataPath', help='Path within object metadata to place new metadata, e.g., path/to/metadatata can later be accessed modelMetadata["path"]["to"]["metadata"]')
    parser.add_argument('--makeSkipped', type=int, help='Reduce framerate by a factor')
    parser.add_argument('--extract', help='Key in metadata to extract, which is saved in --out')
    parser.add_argument('--out', help='Path to save object with updated metadata')
    parser.add_argument('--makeData', action='store_true', help='Make metadata specified by metadataPath be the primary data (does not apply to models)')
    parser.add_argument('--print', help='Key in metadata to print')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    from rPPG.utils import npio

    if Path(args.object).suffix != '.model':
        obj, metadata = npio.loadMeta(args.object)
        if args.metadata and args.metadataPath:
            metadata.embedMetadata(args.metadataPath, args.metadata)
        if args.makeSkipped:
            metadata = metadata.makeSkipped(args.makeSkipped)
        if args.makeData:
            obj = metadata.getPath(args.metadataPath)
        if args.out and not args.extract:
            npio.save(obj, metadata, args.out)
    else:
        from rPPG.utils import modelLoader
        weights, optState, metadata = modelLoader.loadModelData(args.object)
        if args.metadata and args.metadataPath:
            metadata.embedMetadata(args.metadataPath, args.metadata)
        if args.out and not args.extract:
            modelLoader.saveModelData(weights, optState, metadata, args.out)

    if args.extract and args.out:
        newMeta = Metadata()
        newMeta.setFPS(metadata.fps())
        npio.save(metadata[args.extract], newMeta, args.out)

    # Regardless, print metadata keys:
    print(f'Metadata keys: {metadata.keys()}')
    if args.print:
        print(f'Metadata[{args.print}]: {metadata[args.print]}')
