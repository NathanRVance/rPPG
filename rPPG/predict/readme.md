# Predict a waveform based on a cropped video

This step ingests a cropped video in the form of a .npz array and produces a pulse waveform in the form of a .npz array.

The script predict.py is used to perform the predictions. Note that it requires the pretrained model weights. Execute predict.py as follows:

```shell
$ python predict.py video.npz output.npz model
```

where video.npz is the cropped and scaled video produced by [crop.py](../crop/), output.npz is the waveform calculated by the model, and model is the pretrained model to be used.
