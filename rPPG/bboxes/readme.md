# Generate bounding boxes for a video

This step ingests video and produces a .npz array containing timestamped bounding box coordinates.

## Bounding box calculation

We support bounding boxes calculated through MediaPipe, OpenFace, or manually.

### MediaPipe

The preferred method for generating face bounding boxes is to use [MediaPipe](https://github.com/google/mediapipe) for bounding box generation, though the results are not as stable as OpenFace. To use MediaPipe, execute:

```shell
$ python bboxesMp.py video.mp4 output.npz
```

### OpenFace

The [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) tool can be compiled for local use, which is out of scope for this readme.

Use the following command to process videos using OpenFace:

```shell
$ FeatureExtraction -f /path/to/video -out_dir /path/to/results -of resultsName -2Dfp
```

This will generate a csv landmarks file at `/path/to/results/resultsName.csv`.

### Manual

It is also possible to use manually generated bounding boxes. The code for an OpenCV based annotation tool is included in annotator/

## Bounding box formatting

OpenFace and Manually generated landmarks need to be formatted correctly as bounding boxes (this step is not necessary for MediaPipe). The formatBBoxes.py script can be used to convert to a standard format as follows:

```shell
$ python formatBBoxes.py input.csv output.npz
```

where input.csv is the csv file produced by OpenFace or by the manual annotator tool, and output.npz are the bounding boxes produced by this script.
