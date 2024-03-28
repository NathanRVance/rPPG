#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import random

def genSplits(subIDs: list, ratios: dict = {'train': 60, 'val': 20, 'test': 20}) -> dict:
    '''Generates train, val, and test splits based on given ratios

    Args:
      subIDs: list of all subject IDs in the set
      ratios: target ratios dict formatted {splitName: prevalence}

    Returns:
      dict: The splits with same keys as in ratios dict
    '''
    splits = {}
    lenSubIDs = len(subIDs)
    for name, num in ratios.items():
        splits[name] = random.sample(subIDs, k=int(lenSubIDs * num / sum(ratios.values())))
        subIDs = [subID for subID in subIDs if subID not in splits[name]]
    # Distribute the leftovers
    splitNames = list(ratios.keys())
    for i, subID in enumerate(subIDs):
        splits[splitNames[i%len(splitNames)]].append(subID)
    return splits

def genKFolds(subIDs: list, k: int) -> list:
    '''Generates k-folds splits

    Args:
      subIDs: list of all subject IDs in the set
      k: number of folds to generate, k >= 3
    
    Returns:
      list: List of dicts, each containing "train", "val", and "test" splits
    '''
    assert k >= 3
    # Split into k chunks
    random.shuffle(subIDs)
    chunks = [subIDs[i::k] for i in range(k)]
    folds = [{'test': chunks[i], 'val': chunks[(i+1)%k], 'train': sum(list(chunks[(i+j)%k] for j in range(2, k)), [])} for i in range(k)]
    return folds

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Generate train/test/val splits based on available subjects', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('data', help='Directory containing subID.npz formatted data')
    parser.add_argument('out', help='Output splits file, or directory if used with --kfolds >= 3')
    parser.add_argument('--splitRatios', nargs=3, type=float, help='Arguments are train, val, test ratios', default=[60, 20, 20], metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--kfolds', type=int, help='Number of folds to generate. Each subID will be in val and test exactly once; overrides --splitRatios', default=0)
    parser.add_argument('--subjectRegex', default='.*', help='Applied to the filename stem to determine subject ID for subject-disjoint splits')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    import json
    from pathlib import Path
    import re
    subRegex = re.compile(args.subjectRegex)

    stems = [fname.stem for fname in Path(args.data).iterdir()]
    subIDs = list(set(subRegex.search(stem)[0] for stem in stems))

    def expandSplits(splits): # Expands the split contents from subIDs back to stems
        return {key: [stem for stem in stems if subRegex.search(stem)[0] in split] for key, split in splits.items()}

    if args.kfolds:
        folds = genKFolds(subIDs, args.kfolds)
        folds = [expandSplits(fold) for fold in folds]
        Path(args.out).mkdir(exist_ok=True, parents=True)
        for k, fold in enumerate(folds):
            with open(f'{args.out}/{k}-fold.json', 'w') as f:
                json.dump(fold, f)
    else:
        splits = genSplits(subIDs, {'train': args.splitRatios[0], 'val': args.splitRatios[1], 'test': args.splitRatios[2]})
        splits = expandSplits(splits)
        with open(args.out, 'w') as f:
            json.dump(splits, f)
