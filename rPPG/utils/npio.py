#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import json
import h5py
from rPPG.utils import metadata

# Set up a json encoder that handles numpy arrays
from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def save(data: np.ndarray, meta: dict, fname: str):
    """Save data with bundled metadata

    Only one instance of "data" is saved - anything in meta['data'] is omitted.

    Args:
      data: The data to save
      metadata: The metadata to save with it, as metadata.Metadata
      fname: The location to save the data and metadata
    """
    meta = {key: val for key, val in meta.items() if key != 'data'}
    suffix = Path(fname).suffix
    if suffix == '.npz':
        np.savez_compressed(fname, {'data': data, 'metadata': dict(meta)})
    elif suffix == '.npy':
        np.save(fname, {'data': data, 'metadata': dict(meta)})
    elif suffix == '.hdf5' or suffix == '.h5':
        with h5py.File(fname, 'w') as f:
            #f.create_dataset('data', data=data, compression='gzip')
            #f.create_dataset('data', data=data, compression='lzf')
            f.create_dataset('data', data=data)
            f.attrs['metadata'] = json.dumps(meta, cls=NumpyArrayEncoder)
    else:
        raise ValueError(f'Unknown suffix {suffix} on file name {fname}')

def loadMeta(fname: str) -> (np.ndarray, dict):
    """Load the array and its metadata

    This function is smart enough to load data with extensions npz, npy, or hdf5.
    
    If there is no metadata (i.e., it doesn't follow the format of the save function),
    then the raw contents are returned as the np.ndarray and an empty dict is returned
    as the metadata.

    Args:
      fname: The path to the file to load

    Returns:
      np.ndarray: The data that has been loaded
      dict: The metadata
    """
    suffix = Path(fname).suffix
    if suffix == '.npy':
        data = np.load(fname, allow_pickle=True)
        meta = {}
    elif suffix == '.npz':
        with np.load(fname, allow_pickle=True) as data:
            data = data['arr_0']
            meta = data.item()['metadata']
            data = np.array(data.item()['data'])
    elif suffix == '.hdf5' or suffix == '.h5':
        f = h5py.File(fname, 'r')
        data = f['data']
        meta = json.loads(f.attrs['metadata'])
    else:
        raise ValueError(f'Unknown suffix {suffix} on file name {fname}')
    return data, metadata.Metadata(meta)

def load(fname: str) -> np.ndarray:
    """Convenience function to load ignoring (discarding) metadata"""
    return loadMeta(fname)[0]

def meta(fname: str) -> dict:
    """Convenience function to load data stored under "data" in metadata"""
    data, meta = loadMeta(fname)
    meta.setData(data)
    return meta

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Dump an object to stdout', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Object to dump')
    parser.add_argument('--meta', action='store_true', help='Also dump metadata on new line')
    parser.add_argument('--saveFrame', help='Path to save a frame (video data)')
    parser.add_argument('--frameNumber', type=int, default=0, help='Frame number to save')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    obj, meta = loadMeta(args.input)
    print(obj)
    if args.meta:
        print(meta)

    if args.saveFrame:
        import cv2
        cv2.imwrite(args.saveFrame, obj[args.frameNumber])

