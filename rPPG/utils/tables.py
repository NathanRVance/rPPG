#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import numpy as np

def ci(values, confidence=0.95, stddev=False):
    """ Calculate the CI
    Arguments:
      values: A list of values over which to calculate CI
      confidence: Percent confidence for the CI calculation
      stddev: If true, report stddev instead of CI
    """
    if stddev:
        return np.std(values)
    import scipy.stats as st
    if len(values) < 2 or all(val == values[0] for val in values):
        return 0 # CI is undefined for one value or when all values are the same
    interval = st.t.interval(confidence, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))
    return (interval[1]-interval[0])/2

def printTable(data: dict, header: list = None, delimiter: str = '  ', floatPrecision: int = 3, end: str = '', transpose: bool = False, boldThresholds: list = [None, None], boldChars: list = ['\033[1m', '\033[0m']):
    """ Print a nicely formatted table, optionally specifying a header order

    Arguments:
      data: Dictionary of {header: values}
      header: Force a specific order, otherwise is list(data.keys())
      delimiter: String placed between columns
      floatPrecision: Values after the decimal point
      end: String to print at the end of a line
      transpose: If True, header values are printed in the first column
      boldThresholds: Apply bold chars to values <= boldThresholds[0] or >= boldThresholds[1]
      boldChars: Used to surround bolded text
    """
    if not header:
        header = list(data.keys())
    if type(data[header[0]]) is not list: # Unify formatting
        data = {h: [v] for h, v in data.items()}
    # Convert to 2d array
    data = [[h, *data[h]] for h in header]
    if transpose:
        data = list(zip(*data))
    def toStr(value):
        if isinstance(value, (np.floating, float)):
            valuep = f'{value:.{floatPrecision}f}' # Apparenly allows nesting!
            if (boldThresholds[0] is not None and value <= boldThresholds[0]) or (boldThresholds[1] is not None and value >= boldThresholds[1]):
                valuep = boldChars[0] + valuep + boldChars[1]
            value = valuep
        return str(value)
    widths = [max(len(toStr(v)) for v in d) for d in data]
    data = list(zip(*data))
    for row in data:
        print(delimiter.join(f'{toStr(v):<{w}}' for v, w in zip(row, widths)) + end)

def mergeTablesNoCIs(datas: list) -> dict:
    ''' Merge data without calculating confidence intervals

    Arguments:
      datas: List of data dicts formatted {header: values}

    Returns:
      dict: A dictionary formatted {header: [v1, v2, ...]}
    '''
    header = list(datas[0].keys())
    # First fix formatting
    datas = [d if type(d[header[0]]) is list else {h: [v] for h, v in d.items()} for d in datas]
    # Merge into one
    return {h: [[d[h][line] for d in datas] for line in range(len(datas[0][h]))] for h in header if all(h in d for d in datas)}

def mergeTables(datas: list, floatPrecision: int = 3, pm: str = '±', alpha: float = 0.95, stddev=False) -> dict:
    """ Merge data tables with confidence intervals before printing

    Arguments:
      datas: List of data dicts formatted {header: values}
      floatPrecision: Number of digits after the decimal point
      pm: String used for plus/minus when displaying confidence intervals
      alpha: Percent used for confidence interval calculation
      stddev: Use stddev instead of CI

    Returns:
      dict: A suitable input for printTable
    """
    # Now convert to CIs
    data = mergeTablesNoCIs(datas)
    data = {h: [f'{np.mean(vals):.{floatPrecision}f}{pm}{ci(vals, alpha, stddev):.{floatPrecision}f}' for vals in values] for h, values in data.items()}
    return data

def plot(results, columns, plotTitle, out, columnsOverride = None, namesOverride = None, plotType='box', xrotation=0, CI=.95, stddev=False, xlabel=None, ylabel=None, forceIntegerAxis=[], fontFamily=None, lineShowPoints=False, yLims=None, figsize=[6.4,4.8], legendOutside=False, numLegendColumns=1):
    ''' Generate a plot for the data

    Arguments:
      results: The data to plot
      columns: The data columns to plot
      plotTitle: The title for the plot
      out: The path to save the plot
      columnsOverride: Override the names of the columns
      namesOverride: Override the data filenames
      plotType: Either "box" or "line" or "bars". If line, then namesOverride is x and columns are plotted.
      xrotation: Rotation for x ticks
      CI: Confidence interval to plot
      stddev: Use stddev instead of CI
      xlabel: Label for the x axis
      ylabel: Y label when plotting
      forceIntegerAxis: Force integers for axis, any of x, y
      lineShowPoints: Show points rather than CIs when plotting a line
      yLims: Enforce y limits
      figsize: Figsize when plotting
      legendOutside: Position the legend outside of the plot
      numLegendColumns: Number of legend columns
    '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    if fontFamily:
        import matplotlib.font_manager
        plt.rcParams['font.family'] = [fontFamily]
    plt.figure(figsize=figsize)
    for ax, name in [(plt.gca().xaxis, 'x'), (plt.gca().yaxis, 'y')]:
        if name in forceIntegerAxis:
            ax.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.ticklabel_format(style='plain', axis=name, useOffset=False)
    plt.title(plotTitle)
    if not columnsOverride:
        columnsOverride = columns
    if not namesOverride:
        namesOverride = ['' for res in results]
    if len(columnsOverride) == 1:
        labels=namesOverride
        plt.ylabel(columnsOverride[0])
    else:
        labels=[f'{n} {c}' for n in namesOverride for c in columnsOverride]
    if plotType == 'box':
        x = []
        for res in results:
            res = mergeTablesNoCIs(res)
            x += [res[c][0] for c in columns]
        plt.boxplot(x, labels=labels)
    elif plotType == 'line':
        x = [float(n) for n in namesOverride]
        ys = {c: [] for c in columns}
        for res in results:
            res = mergeTablesNoCIs(res)
            for c in columns:
                ys[c].append(res[c][0])
        for label, data in ys.items():
            plt.plot(x, [np.mean(d) for d in data], label=label)
            if len(data[0]) > 1:
                if lineShowPoints:
                    xExp = [xVal for xVal, yVals in zip(x, data) for _ in yVals]
                    yFlat = [yVal for yVals in data for yVal in yVals]
                    plt.scatter(xExp, yFlat)
                else:
                    CIs = [(np.mean(d)-ci(d, CI, stddev), np.mean(d)+ci(d, CI, stddev)) for d in data]
                    plt.fill_between(x, [CI[0] for CI in CIs], [CI[1] for CI in CIs], alpha=0.5)
    elif plotType == 'bars':
        width=1/(len(columns)+1)
        results = [mergeTablesNoCIs(res) for res in results]
        for colIndex, (c, coverride) in enumerate(zip(columns, columnsOverride)):
            errs = [ci(res[c][0], CI, stddev) for res in results]
            plt.bar(np.arange(len(namesOverride))+width*colIndex, [np.mean(res[c][0]) for res in results], width, yerr=errs, label=coverride)
        plt.gca().set_xticks(np.arange(len(namesOverride))+width, namesOverride)
    else:
        raise ValueError(f'Unknown plot type {plotType}')
    if plotType == 'line' or plotType == 'bars': # May want a legend
        if len(columns) > 1:
            loc='best'
            bbox_to_anchor=plt.gcf().bbox
            if legendOutside:
                if numLegendColumns == 1:
                    bbox_to_anchor=(1,1)
                    loc='upper left'
                else:
                    bbox_to_anchor=(0, 1.02, 1, 0.2)
                    loc='lower left'
                plt.legend(ncols=numLegendColumns, bbox_to_anchor=bbox_to_anchor, loc=loc, mode='expand')
            else:
                plt.legend(ncols=numLegendColumns)
        else:
            plt.ylabel(columnsOverride[0])
    if yLims is not None:
        plt.gca().set_ylim(yLims)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xrotation != 0:
        plt.xticks(rotation=xrotation, ha='right')
    plt.tight_layout()
    plt.savefig(out)
    #plt.show()

if __name__ == '__main__':
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description='Format testing results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results', nargs='+', help='json-formatted results file(s) to display. Will compute CIs if multiple results files are given')
    parser.add_argument('--moreResults', nargs='+', action='append', default=[], help='Additional rows')
    parser.add_argument('--ci', type=float, default=0.95, help='Confidence interval percent')
    parser.add_argument('--stddev', action='store_true', help='Report stddev rather than CIs')
    parser.add_argument('--format', choices=['plain', 'latex'], default='plain', help='Formatting to use')
    parser.add_argument('--underscores', action='store_true', help='When formatting latex, escape underscores. Default uses math mode and places everything after the underscore as subscript.')
    parser.add_argument('--columns', nargs='+', help='Columns to include. By default uses all columns')
    parser.add_argument('--transpose', action='store_true', help='Output a transposed table with headers on the first column')
    parser.add_argument('--abs', action='store_true', help='Take the absolute value of columns before combining')
    parser.add_argument('--plot', help='Plot the data as a boxplot, with one box per column given in --columns. Requires multiple results files.')
    parser.add_argument('--plotType', default='box', choices=['box', 'line', 'bars'], help='type of plot to plot. If line, requires numerics for plotMoreResultsNames.')
    parser.add_argument('--forceIntegerAxis', choices=['x', 'y'], default=[], nargs='+', help='Force integers for axis, any of x, y')
    parser.add_argument('--xticksRotation', type=float, default=0, help='Set x tick rotation when plotting')
    parser.add_argument('--lineShowPoints', action='store_true', help='When pltType is line, show points rather than CIs')
    parser.add_argument('--yLims', type=int, nargs=2, help='Enforce y limits when plotting')
    parser.add_argument('--plotTitle', help='Title for the boxplot')
    parser.add_argument('--plotFontFamilyOverride', help='Override the font family with, e.g., "serif"')
    parser.add_argument('--smallplot', action='store_true', help='Make figsize small for relatively larger plot text (overrides figsize, setting to [4,3]')
    parser.add_argument('--figsize', default=[6.4,4.8], type=float, nargs=2, help='Figsize when plotting')
    parser.add_argument('--plotColumnNamesOverride', nargs='+', help='Override column names when plotting')
    parser.add_argument('--plotMoreResultsNames', nargs='+', help='Names given to results if --moreResults')
    parser.add_argument('--plotYlabel', help='Y axis label when plotting')
    parser.add_argument('--legendOutside', action='store_true', help='Position the legend outside of the plot')
    parser.add_argument('--numLegendColumns', type=int, default=1, help='Number of legend columns when plotting')
    parser.add_argument('--moreResultsColumnName', default='Name', help='Column name when --plotMoreResultsNames is given, also used for plot x axis label')
    parser.add_argument('--boldAbove', type=float, default=None, help='Threshold to bold results')
    parser.add_argument('--boldBelow', type=float, default=None, help='Threshold to bold results')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.smallplot:
        args.figsize=[4,3]

    import json

    def load(fname):
        with open(fname) as f:
            res = json.load(f)
            if args.abs:
                for h in res:
                    res[h] = abs(res[h])
            return res


    results = [[load(fname) for fname in fnames] for fnames in [args.results] + args.moreResults]
    # Fix results: Should be 2d list of dicts, but may sometimes have 3d lists embedded
    def flatten(lol):
        flat = []
        if not lol:
            return flat
        for i in lol:
            if type(i) == list:
                flat.extend(flatten(i))
            else:
                flat.append(i)
        return flat
    results = [flatten(lol) for lol in results]
    columns = args.columns if args.columns else list(results[0][0].keys())

    pm = '±'
    delimiter = '  '
    end = ''
    boldChars = ['\033[1m', '\033[0m']
    if args.format == 'latex':
        def handleUnderscore(key: str) -> str:
            if '_' not in key:
                return key
            if args.underscores:
                return key.replace('_', '\_')
            else:
                import re
                return '$' + re.sub(r'_(.*)', r'_{\1}', key) + '$'
        pm = ' $\\pm$ '
        delimiter = ' & '
        end = ' \\\\'
        boldChars = ['\\textbf{', '}']
        for res in results:
            for r in res:
                for key in list(r.keys()):
                    if handleUnderscore(key) != key:
                        r[handleUnderscore(key)] = r[key]
                        del r[key]
        columns = [handleUnderscore(key) for key in columns]
        numcols = len(columns)
        if args.transpose:
            # Do rows instead
            numcols = len(results)
        if args.plotMoreResultsNames:
            numcols += 1
        print(f'\\begin{{tabular}}{{{"c"*numcols}}}\n\\toprule')

    resultsToPlot = []
    tableCols = columns
    if args.plotMoreResultsNames:
        tableCols = [args.moreResultsColumnName] + tableCols
    resultsToPrint = {c: [] for c in tableCols}
    for i, res in enumerate(results):
        if len(res) == 1:
            res = res[0]
        else:
            resultsToPlot.append(res)
            res = mergeTables(res, pm=pm, alpha=args.ci, stddev=args.stddev)
        if args.plotMoreResultsNames:
            res[args.moreResultsColumnName] = [args.plotMoreResultsNames[i]]
        for col in tableCols:
            try:
                resultsToPrint[col].append(res[col][0])
            except:
                resultsToPrint[col].append(res[col])
    printTable(resultsToPrint, header=tableCols, delimiter=delimiter, end=end, transpose=args.transpose, boldThresholds=[args.boldBelow, args.boldAbove], boldChars=boldChars)
    if args.plot:
        plot(resultsToPlot, columns, args.plotTitle, args.plot, columnsOverride = args.plotColumnNamesOverride, namesOverride = args.plotMoreResultsNames, plotType=args.plotType, xrotation=args.xticksRotation, CI=args.ci, stddev=args.stddev, xlabel=args.moreResultsColumnName, ylabel=args.plotYlabel, forceIntegerAxis=args.forceIntegerAxis, fontFamily=args.plotFontFamilyOverride, lineShowPoints=args.lineShowPoints, yLims=args.yLims, figsize=args.figsize, legendOutside=args.legendOutside, numLegendColumns=args.numLegendColumns)

    if args.format == 'latex':
        print(f'\\bottomrule\n\\end{{tabular}}')
