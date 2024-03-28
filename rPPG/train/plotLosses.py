#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import statistics

def plotLosses(losses: list, title: str = 'Train and Val Losses', appendLabel: str = '', CI: float = 0.95, skipCI: bool = False, plt = None, columns: list = ['Loss-Train', 'Loss'], columnNames: list = [], ylabel: str = 'Loss'):
    """Plot losses using matplotlib

    Arguments:
      losses: Loss data as a list, with each element being the contents of results_avg.json
      appendLabel: Append string to labels, which are 'train' and 'val'
      CI: Confidence interval (default 0.95)
      skipCI: Skips plotting the CI
      plt: Plot over an existing plot rather than create a new one
      columns: List of columns to plot
      columnNames: Labels to use for columns (default to contents of columns)

    Returns:
      plt: The matplotlib.pyplot that was created
    """
    if not columnNames:
        columnNames = columns
    from rPPG.utils.tables import ci
    if not plt:
        import matplotlib.pyplot as plt
        #plt.figure(figsize=(4,3))
        plt.figure()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
    # Trim losses based on columns
    losses = [[l for l in loss if all(c in l for c in columns)] for loss in losses]
    lss = {n: [[e[c] for e in epoch] for epoch in zip(*losses)] for c, n in zip(columns, columnNames)}
    #for epoch in zip(*losses):
    #    for c in columns:
    #        lss[c].append([e[c] for e in epoch])

    x = list(range(len(list(lss.values())[0])))

    for name, l in lss.items():
        plt.plot(x, [statistics.mean(y) for y in l], label=f'{name}{appendLabel}')
        if len(l[0]) > 1 and not skipCI:
            CIs = [(statistics.mean(y) - ci(y, CI), statistics.mean(y) + ci(y, CI)) for y in l]
            # Print diagnostic info about CIs
            print(f'Median CI for {name}{appendLabel}: ±{statistics.median([ci(y, CI) for y in l])}')
            plt.fill_between(x, [CI[0] for CI in CIs], [CI[1] for CI in CIs], alpha=0.5)
    plt.legend()
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Plot train and valitaion losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('losses', nargs='+', help='json-formatted loss file(s) to plot. Will plot CIs if multiple loss files are given.')
    parser.add_argument('--out', help='Path to plot file to save')
    parser.add_argument('--columns', nargs='+', help='Columns to plot', default=['Loss-Train', 'Loss'])
    parser.add_argument('--columnNamesOverride', nargs='+', help='Override names of colunms to plot', default=[])
    parser.add_argument('--mlosses', action='append', nargs='+', help='More loss files to plot.', default=[])
    parser.add_argument('--mlossesNames', nargs='+', help='Names of losses input (by default are numbered)', default=[])
    parser.add_argument('--ci', type=float, default=0.95, help='Confidence interval percent')
    parser.add_argument('--skipCI', action='store_true', help='When given multiple loss files, plot average loss only, omitting CIs.')
    parser.add_argument('--title', help='Title for the plot', default='Train and Val Losses')
    parser.add_argument('--ylabel', help='Y label for the plot', default='Loss')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    import json
    plt=None
    args.mlosses = [args.losses] + args.mlosses
    if not args.mlossesNames:
        args.mlossesNames = list(range(len(args.mlosses)))
    for name, ml in zip(args.mlossesNames, args.mlosses):
        losses = []
        for lf in ml:
            with open(lf) as f:
                losses.append(json.load(f))
        plt = plotLosses(losses, title=args.title, appendLabel='' if len(args.mlosses) == 1 else f' {name}', CI=args.ci, skipCI=args.skipCI, plt=plt, columns=args.columns, columnNames=args.columnNamesOverride, ylabel=args.ylabel)
    
    plt.savefig(args.out)
