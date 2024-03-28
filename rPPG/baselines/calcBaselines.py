#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from rPPG.utils import npio

def loadAndCalculate(signals: str = None, video: str = None, bboxes: list = None) -> (list, list):
    """Loads signals or video, then calculates CHROM and POS

    If neither signals nor video are provided then a ValueError is raised

    Args:
      signals: Path to saved signals
      video: Path to video
      bboxes: Boxes to feed into baselinesPreprocess.calcSignals
    
    Returns:
      (CHROM, POS, metadata) results
    """
    from rPPG.utils import metadata
    from rPPG.baselines import CHROM, POS, baselinesPreprocess

    if signals:
        signals, meta = npio.loadMeta(signals)
        if not meta and video:
            meta = metadata.Metadata.fromVideo(video)
    else:
        if video:
            meta = metadata.Metadata.fromVideo(video)
            signals = baselinesPreprocess.calcSignals(video, bboxes=bboxes)
        else:
            raise ValueError('Must provide signals or video')

    return CHROM.process_CHROM(signals, meta.fps(), windowed=False), POS.process_POS(signals, meta.fps()), meta

if __name__ == '__main__':
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description='Compute CHROM and POS waveforms', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--signals', help='Path to spatially averaged video signals')
    parser.add_argument('--video', help='Path to the input video')
    parser.add_argument('--CHROM', help='Path to save CHROM waveform')
    parser.add_argument('--POS', help='Path to save POS waveform')
    parser.add_argument('--saveCombined', help='Path to save combined waveform')
    parser.add_argument('--saveCombinedAppend', nargs='+', help='Append additional waves when saving combined')
    parser.add_argument('--saveCombinedLength', type=int, default=9, help='Number of waves to combine (padded with averaged waves)')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    CHROM, POS, meta = loadAndCalculate(args.signals, args.video)
    if args.CHROM:
        npio.save(CHROM, meta, args.CHROM)
    if args.POS:
        npio.save(POS, meta, args.POS)
    if args.saveCombined:
        combined = [CHROM, POS]
        for append in args.saveCombinedAppend:
            combined.append(npio.load(append))
        # Make all the same length
        minlen = min(len(c) for c in combined)
        combined = [c[:minlen] for c in combined]
        import numpy as np
        # Also make all normalized
        combined = [(c-np.mean(c))/np.std(c) for c in combined]
        while len(combined) < args.saveCombinedLength:
            combined.append(np.mean(combined, axis=0))
        combined = np.column_stack(combined)
        npio.save(combined, meta, args.saveCombined)
