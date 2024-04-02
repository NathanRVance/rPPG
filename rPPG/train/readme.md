# Training Code

## Configuration File

The json-formatted configuration file contains the hyperparameters used both for the model architecture and training. An example configuration is given in [exampleConfig.json](exampleConfig.json). The configuration is read into a [Config object](../utils/Config.py).

The parameters for the model are as follows:

 * architecture: Any of 'CNN3D', 'NRNet', 'NRNet\_simple', 'Flex', 'TS-CAN', 'litenetv2', 'litenetv6'
 * depth (Flex only): The depth in layers of the model
 * padding\_mode (Flex only): Any padding\_mode supported by [torch.nn.Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
 * hidden\_channels (Flex only): Hidden layer channels
 * flex\_dilation (Flex only): Perform dilation in flex models
 * tk: The temporal kernel width
 * channels: The input channels and order; any combination of characters from 'n', 'rgb', 'hsv', 'yuv'
 * out\_channels: If 1, then output is shape [B,T]. If >1, then output is shape [B,T,C].
 * num\_waves: Used in NRNet\_simple, number of waves for noise reduction.
 * frame\_width and frame\_height: The width and height in pixels of the input cropped and scaled video.
 * fpc: The number of frames per clip as fed into the 3dcnn in each evaluation of the network.
 * fps: Framerate of the training data.

The parameters for training are as follows:

 * num\_workers: The number of threads dedicated to preprocessing and feeding data.
 * num\_epochs: The number of epochs for which to train the model.
 * augmentation: Set of augmentations performed on the training data. Any combination of characters from 'figscsmn' standing for flipping, illumination, Gaussian noise, speed, crop, modulate, negative.
 * negative\_probability: Used by negative augmentation as probability to generate negative sample.
 * noise\_width: Amount of noise used by negative augmentation.
 * normalization: Any of 'scale', 'stddev', 'histogram'
 * dropout: Dropout used in the model.
 * batch\_size: Batch size for training.
 * lr: Learning rate.
 * masks: Whether to utilize masks if bundled with the video.
 * loss: Dict where keys are 'negpearson', 'mae', 'mse', 'mcc', 'envelope', 'bandwidth', 'sparsity', 'variance'; and values are relative weight (floats).
 * negativeLoss: Dict where keys are 'deviation', 'specentropy', 'specflatness'; and values are relative weight (floats).

Finally, there are parameters for evalution:

 * val\_test\_fpc: Defines a fpc for validation and test sets (default: use same fpc as training). If set to 0 uses full clip.
 * hr\_method: Method for calculating heart rate from the waveform, any of 'fft', 'peaks', 'cwt', 'sssp'
 * hz\_low: Minimum bound on heartrate, in hz.
 * hz\_high: Maximum bound on heartrate, in hz.
 * fft\_window: If using fft, the window size, in seconds, over which to calculate the fft.
 * skip\_fft: Do not use fft in evaluation to calculate SNR. Sometimes necessary for large datasets.
 * move\_cost: Used by sssp to weight cost to jump in HR.
 * smooth\_method: Method for performing smoothing, either smooth or delta\_limit.
 * delta\_limit: If using delta\_limit for smooth\_method, maximum change in heartrate, in bpm/second, allowed.
 * smooth: If using smooth for smooth\_method, window size, in seconds, for smoothing.
 * multiprocessing: Use a thread pool when evaluating.
 * columns: List of evaluation metrics columns, default prints all.

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
$ ./test.py path/to/model path/to/gt/ path/to/videos/
```

This prints evaluation metrics in a tabular format. Common metrics include:
 * ME: mean error
 * MAE: mean absolute error
 * RMSE: root mean square error
 * r\_HR: Pearson r correlation between the ground truth and predicted heart rate
 * r\_wave: Pearson r correlation between the ground truth and predicted waves
 * MXCorr: Maximum cross correlation between the ground truth and predicted waves

If test.py is executed with the `--save` flag, then an output directory will be generated with the following contents:

 * `results_bySubject.json` contains subject-wise validation metrics
 * `results_avg.json` contains validation metrics averaged over all subjects and weigthed by length of subject video
 * `gt_waves/` contains the ground truth wave for each subject
 * `waves/` contains the predicted wave for each subject
 * `gt_hr/` contains the ground truth heart rate for each subject
 * `hr/` contains the predicted heart rate for each subject
