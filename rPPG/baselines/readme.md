# Baseline color transformation methods for rPPG.

Since both CHROM and POS are lightweight and require the same inputs, we can run both at the same time.

```
python calcBaselines.py --video path/to/video.mp4 --CHROM path/to/save/CHROM.npz --POS path/to/save/POS.npz
```

## References
**Plane-Orthogonal-to-Skin (POS)**
https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf

**Chrominance (CHROM)**
https://www.es.ele.tue.nl/~dehaan/pdf/169_ChrominanceBasedPPG.pdf

