# Crop video based on bounding boxes

This step ingests a .npz array containing bounding boxes and a video file, and produces a cropped video in the form of a .npz array.

Execute crop.py as follows:

```shell
$ python crop.py video.mp4 boxes.npz output.npz
```

where video.mp4 is a video file in any format that is supported by OpenCV, boxes.npz is the output from [formatBBoxes.py](../bboxes/), and output.npz is the cropped and scaled video which will be input to the model. A `--debug` flag may optionally be used to produce a debugging video, which will be saved as output.npz-debug.avi.
