#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

def getBestEpoch(results: list, optimize: str = 'Loss') -> int:
    """Get the best training epoch
    
    Arguments:
      results: Training results as saved in results_avg.json
      optimize: Metric to optimize when finding best epoch (default is Loss)

    Returns:
      The index of the best epoch
    """
    return min(range(len(results)), key=lambda i: results[i][optimize])


if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Get the best model based on training results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results', help='Path to training results_avg.json')
    parser.add_argument('--optimizeMetric', default='Loss', help='Metric to optimize in model selection')
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from pathlib import Path
    results = Path(args.results)
    import json
    with results.open() as f:
        bestIndex = getBestEpoch(json.load(f), optimize=args.optimizeMetric)
    bestModel = results.parent / f'rpnet_e{bestIndex}'
    print(bestModel)
