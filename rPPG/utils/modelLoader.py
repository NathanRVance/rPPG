#!/usr/bin/env python3
import torch
import json
from rPPG.utils import config as cfg
from rPPG.utils import metadata as md

def getCudaDevice(minFreeMem: int = 2048) -> torch.device:
    """Selects the cuda device with the most free memory, or cpu as a fallback
    
    Args:
      minFreeMem: Minimum amount of free memory on device in MB (default: 2048 MB)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadModelData(modelPath: str, device: torch.device = torch.device('cpu')) -> (dict, dict, dict):
    """Bare-bones data loading obtaining data associated with a model

    Args:
      modelPath: Location of the model on disk
      device: (optional) device to map data to

    Returns:
      dict: Model weights
      dict: Optimizer state dict, or None if not available
      dict: Metadata (aka, "config"), or empty dict if not available
    """
    metadata = {}
    optimizerStateDict = None
    modelDict = torch.load(modelPath, map_location=device)
    if 'config' in modelDict:
        metadata = modelDict['config']
    if 'optimizer_state_dict' in modelDict:
        optimizerStateDict = modelDict['optimizer_state_dict']
    return modelDict['model_state_dict'], optimizerStateDict, md.Metadata(metadata)

def saveModelData(weights: dict, optimizerStateDict: dict, metadata: dict, fname: str):
    """Save model weights with bundled metadata

    Args:
      weights: the weigts to save
      optimizerStateDict: save optimizer state, or None
      metadata: Metadata to save, or None
      fname: The location to save the model data
    """
    modelDict = {'model_state_dict': weights}
    if optimizerStateDict:
        modelDict['optimizer_state_dict'] = optimizerStateDict
    if metadata:
        modelDict['config'] = dict(metadata)
    torch.save(modelDict, fname)

def load(modelPath: str = None, configPath: str = None, config: dict = {}, architecture: str = None) -> ('model', dict):
    """Load the model from the disk

    Args:
      modelPath: Location of the model (default: create uninitialized model)
      configPath: Location of json-formatted config (default: use config bundled with the model, falling back on default config)
      config: Initial config to be merged with contents of configPath and config bundled with the model
      architecture: Model architecture

    Returns:
      model: The model that was loaded
      dict: The config for the model
    """
    device = getCudaDevice()
    if modelPath:
        weights, optimizerStateDict, conf = loadModelData(modelPath, device)
        if config is None:
            config = {}
        config = cfg.Config(md.mergeDict(config, conf))
    if configPath:
        with open(configPath) as f:

            config = cfg.Config(md.mergeDict(json.load(f), config))
    if architecture == None:
        architecture = config.architecture()
    if architecture == 'Flex':
        from rPPG.utils.models import Flex as modelArch
    elif architecture == 'CNN3D':
        from rPPG.utils.models import CNN3D as modelArch
    elif architecture == 'NRNet':
        from rPPG.utils.models import NRNet as modelArch
    elif architecture == 'NRNet_simple':
        from rPPG.utils.models import NRNet_simple as modelArch
    elif architecture == 'litenetv2':
        from rPPG.utils.litemodels import LiteNet_v2 as modelArch
    elif architecture == 'litenetv6':
        from rPPG.utils.litemodels import LiteNet_v6 as modelArch
    elif architecture == 'TS-CAN':
        from rPPG.utils.TSCAN import TSCAN as modelArch
    else:
        raise ValueError(f'Unknown model architecture {architecture}')
    if not config:
        config = cfg.Config()
    model = modelArch(config)
    model.optimizer_state_dict = None
    if modelPath:
        model.load_state_dict(weights)
        if optimizerStateDict:
            model.optimizer_state_dict = optimizerStateDict
    model = model.float().to(device)
    return model, config

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print model architectures', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', help='Path to saved model')
    parser.add_argument('--config', help='Path to saved config')

    args = parser.parse_args()

    model, config = load(modelPath=args.model, configPath=args.config)
    print(model)
