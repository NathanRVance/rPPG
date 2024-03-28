#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot CKA results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('cka', nargs='+', help='.npy formatted CKA results')
    parser.add_argument('out', help='Path to save plotted output')
    parser.add_argument('--maxPlotsRow', type=int, default=4, help='Maximum number of plots to place in a row')
    parser.add_argument('--title', default='CKA results', help='Title for the plot')
    parser.add_argument('--subtitles', nargs='+', help='Titles for subplots, by default uses file stems')
    parser.add_argument('--ylabel', help='Override y label (otherwise uses model1 name)')
    parser.add_argument('--xlabel', help='Override x label (otherwise uses model2 name)')
    parser.add_argument('--maxAxisTicks', default='auto', help='Maximum number of ticks on each axis')
    parser.add_argument('--transposeIndices', nargs='*', default=[], type=int, help='Indices of CKA results to transpose')
    parser.add_argument('--cmap', default='gray', help='Colormap to use when plotting')
    parser.add_argument('--fontsize', type=float, help='Font size when plotting')

    args = parser.parse_args()

    results = [np.load(fname, allow_pickle=True).item() for fname in args.cka]

    for i in args.transposeIndices:
        results[i]['CKA'] = results[i]['CKA'].T
        tmp = results[i]['model1_name']
        results[i]['model1_name'] = results[i]['model2_name']
        results[i]['model2_name'] = tmp
    
    cols = min(args.maxPlotsRow, len(results))
    rows = int(np.ceil(len(results)/cols))
    figwidth = 7 * (cols+1) / rows
    figheight = figwidth * rows / (cols+1)
    if args.fontsize:
        plt.rcParams.update({'font.size': args.fontsize})
    fig, axs = plt.subplots(nrows=rows, ncols=cols, layout='compressed', figsize=(figwidth, figheight))
    fig.suptitle(args.title)
    
    # Figure out subtitles
    if args.subtitles is None:
        from pathlib import Path
        args.subtitles = [Path(fname).stem for fname in args.cka]
    else:
        while len(args.subtitles) < len(args.cka):
            args.subtitles.append('')

    if len(results) == 1: # fix axs to avoid issues
        axs = np.array([[axs]])

    for res, ax, subtitle in zip(results, axs.flat, args.subtitles):
        r, c = np.where(axs == ax)
        if args.ylabel is None:
            ax.set_ylabel(res['model1_name'])
        elif c == 0:
            ax.set_ylabel(args.ylabel)
        if args.xlabel is None:
            ax.set_xlabel(res['model2_name'])
        elif r == rows-1:
            ax.set_xlabel(args.xlabel)
        #ylabel = args.ylabel if args.ylabel is not None else 'Layers ' + res['model1_name']
        #xlabel = args.xlabel if args.xlabel is not None else 'Layers ' + res['model2_name']
        #ax.set_ylabel(ylabel)
        #ax.set_xlabel(xlabel)
        ax.set_title(subtitle)
        for x, name in [(ax.xaxis, 'x'), (ax.yaxis, 'y')]:
            x.set_major_locator(ticker.MaxNLocator(nbins=args.maxAxisTicks, integer=True))
            plt.ticklabel_format(style='plain', axis=name, useOffset=False)
        imgrows, imgcols = res['CKA'].shape
        im = ax.imshow(res['CKA'], origin='lower', vmin=0, vmax=1, cmap=args.cmap,
                       extent=(0.5, imgcols+0.5, 0.5, imgrows+0.5))

    plt.colorbar(im, ax=axs.ravel().tolist(), label='Similarity')
    plt.savefig(args.out)
