# Training Code

## Configuration File

The json-formatted configuration file contains the hyperparameters used both for the model architecture and training. An example configuration is given in [exampleConfig.json](exampleConfig.json).

The parameters for the model are as follows:

 * tk: The temporal kernel width
 * channels: The input channels and order; any combination of r, g, b, or n.
 * frame\_width and frame\_height: The width and height in pixels of the input cropped and scaled video.
 * fpc: The number of frames per clip as fed into the 3dcnn in each evaluation of the network.
 * step: The step length, in frames, between evaluations of the network.
 * fps: Framerate of the training data.

The parameters for training are as follows:

 * num\_workers: The number of threads dedicated to preprocessing and feeding data.
 * num\_epochs: The number of epochs for which to train the model.
 * augmentation: Set of augmentations performed on the training data. Any combination of f, i, or g standing for flipping, illumination, and Gaussian noise.
 * dropout: Dropout used in the model.
 * batch\_size: Batch size for training.
 * lr: Learning rate.

Finally, there are parameters for evalution:

 * hr\_method: Method for calculating heart rate from the waveform, either fft or peaks.
 * hz\_low: Minimum bound on heartrate, in hz.
 * hz\_high: Maximum bound on heartrate,in hz.
 * fft\_window: If using fft, the window size, in seconds, over which to calculate the fft.
 * smooth\_method: Method for performing smoothing, either smooth or delta\_limit.
 * smooth: If using smooth for smooth\_method, window size, in seconds, for smoothing.
 * delta\_limit: If using delta\_limit for smooth\_method, maximum change in heartrate, in bpm/second, allowed.

## Splits File

The optional json-formatted splits file contains the subject IDs used for the train, val, and test splits. An example for the DDPM dataset is given in [exampleSplits.json](exampleSplits.json).

## Training

The script train.py is used to train the model. Execute train.py as follows:

```shell
$ ./train.py config.json path/to/gt/ path/to/videos/ path/to/output/
```

where config.json is the configuration file detailed above, the path/to/gt/ contains npz formatted ground truth waves with filenames subID.npz and in the same format as the waves produced by [predict.py](../predict/), the path/to/videos/ contains npz formatted videos with filenames subID.npz and in the same format as the cropped and scaled videos produced by [crop.py](../crop/), and path/to/output/ will be created by the script and populated with trained models. The output has a few relevant features:

 * The file output/results\_bySubject.json contains subject-wise validation metrics.
 * The file output/results\_avg.json contains validation metrics averaged over all subjects and weigthed by length of subject video.
 * The file output/splits.json contains the splits used.
 * Every saved model contains a dictionary with keys `model_state_dict` (value is model weights) and `config` (value is a copy of the configuration passed into train.py).

The script can optionally be configured to take a splits file as input, or perform training with random splits. See `train.py --help` for details.

## Testing

The `test.py` script can be used to evaluate a trained model as follows:

```shell
$ ./test.py splits.json path/to/model path/to/gt/ path/to/videos/
```

This outputs evaluation metrics in the following format:

```
Loss, ME, MAE, RMSE, r_HR, r_wave
0.330, -1.082, 1.999, 4.777, 0.808, 0.562
```

In the above, Loss is the loss for the model, ME is the mean error, MAE is mean absolute error, RMSE is root mean square error, r\_HR is the Pearson r correlation between the ground truth and predicted heart rate, and r\_wave is the Pearson r correlation between the ground truth and predicted waves.

If test.py is executed with the `--save` flag, then an output directory will be generated with the following contents:

 * `results_bySubject.json` contains subject-wise validation metrics
 * `results_avg.json` contains validation metrics averaged over all subjects and weigthed by length of subject video
 * `gt_waves/` contains the ground truth wave for each subject
 * `waves/` contains the predicted wave for each subject
 * `gt_hr/` contains the ground truth heart rate for each subject
 * `hr/` contains the predicted heart rate for each subject
