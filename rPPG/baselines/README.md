# Baseline color transformation methods for rPPG.
# Since both CHROM and POS are lightweight and require the same inputs, we can run both at the same time.

```
python CHROM_POS_preprocess.py <video> <landmarks> <signals>
python CHROM_POS_process.py <signals> <framerate> <output>
```

## References
**Plane-Orthogonal-to-Skin (POS)**
https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf

**Chrominance (CHROM)**
https://www.es.ele.tue.nl/~dehaan/pdf/169_ChrominanceBasedPPG.pdf

