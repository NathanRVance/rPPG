from progress.bar import IncrementalBar
from copy import deepcopy
import numpy as np
import torch

def execModel(model, loader, optimizer=None, train: bool = False, barLabel: str = 'Inferring') -> (dict, float):
    """ Infers or trains a model over a dataset

    Args:
      model: The model to execute
      loader: The dataloader
      optimizer: The optimizer for training
      train: If true, trains the model using backpropogation
      barLabel: Printed label on the progress bar

    Returns:
      dict: Predicted results as {subID: list}
      float: Loss, if ground truth is provided

    """
    if train:
        model.train()
    else:
        model.eval()
    device = next(model.parameters()).device
    # Make a progress bar
    suffix = '%(index)d/%(max)d - %(elapsed)d s - loss: %(loss).4f%(lossParts)s'
    bar = IncrementalBar(barLabel, width=10, max=len(loader), suffix=suffix)
    bar.loss = 0.0
    bar.lossParts = ''
    def formatLossParts(lossParts: dict) -> str:
        if len(lossParts) < 2:
            return ''
        def minPrefix(s) -> str:
            for prefix in [s[:i] for i in range(1, len(s)+1)]:
                if not any(name.startswith(prefix) for name in lossParts.keys() if name != s):
                    return prefix
        return ' (' + ' + '.join(minPrefix(name) + f': {lp:.3f}' for name, lp in lossParts.items()) + ')'
    lossPartsSummary = {}
    predictions = {}
    for i, (clip, extras) in enumerate(loader):
        with torch.set_grad_enabled(train):
            if train:
                optimizer.zero_grad()
            outputs = model(clip.to(device))
            if extras['wave'].numel() > 0:
                loss, lossParts = model.loss(outputs, extras)
                bar.loss = (bar.loss * i + loss.item()) / (i+1)
                for lossName, lp in lossParts.items():
                    if not lossName in lossPartsSummary:
                        lossPartsSummary[lossName] = 0.0
                    lossPartsSummary[lossName] = (lossPartsSummary[lossName] * i + lp.item()) / (i+1)
                bar.lossParts = formatLossParts(lossPartsSummary)
                if train:
                    loss.backward()
                    optimizer.step()
            if not train:
                subID = extras['subID'][0]
                if subID not in predictions:
                    predictions[subID] = []
                predictions[subID].append(deepcopy(outputs.cpu().numpy()))
            del outputs
        bar.next()
    bar.finish()
    return predictions, bar.loss, lossPartsSummary
