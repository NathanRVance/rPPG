These models were trained as part of the paper:

Vance, Nathan, and Patrick Flynn. "Refining Remote Photoplethysmography Architectures using CKA and Empirical Methods." arXiv preprint arXiv:2401.04801 (2024).

They achieved the following within-dataset results:

```
Name    ME     MAE    RMSE    MXCorr
DDPM-6  0.281  1.926  4.205   0.536
MSPM-6  5.743  7.752  16.837  0.467
```

MSPM-6 had the following breakdown by segment:

```
Name           ME      MAE     RMSE    MXCorr
all            5.743   7.752   16.837  0.467
adversarial    38.020  38.140  41.030  0.141
unadversarial  0.433   1.929   4.951   0.599
```

Cross-dataset results (including MSPM breakdown):

```
Train  Eval                ME      MAE     RMSE    MXCorr
MSPM   DDPM                -4.859  19.061  28.129  0.246i
MSPM   PURE                1.702   2.481   4.157   0.655
MSPM   UBFC-rPPG           -0.576  1.365   3.245   0.674
DDPM   MSPM-all            5.280   6.985   16.352  0.498
DDPM   MSPM-adversarial    37.776  38.189  41.087  0.120
DDPM   MSPM-unadversarial  -0.063  1.449   3.999   0.720
DDPM   PURE                1.666   2.060   3.682   0.663
DDPM   UBFC-rPPG           -0.089  0.938   2.465   0.725
```
