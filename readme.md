# Code to train and use rPPG models

## Setup

This code is set up as an installable python module. To install in a virtual environment:

```console
python -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install --editable .
```

The `--editable` flag may be omitted if this is a deployment installation.

## Usage
1. Calculate bounding boxes. Code for this task is in [rPPG/bboxes/](rPPG/bboxes/)
2. Crop videos based on bounding boxes. Code for this task is in [rPPG/crop/](rPPG/crop/)
3. Predict the pulse waveform for these cropped videos. Code for this task is in [rPPG/predict/](rPPG/predict/)
4. View and postprocess predictions. Code for this task is in [rPPG/postprocess/](rPPG/postprocess/)

## Model and run configuration
The models have various hyperparameters used for the model architecture, training and evaluation/validation. We save these hyperparameters as a json-formatted dictionary. The complete specification is in [rPPG/train/readme.md](rPPG/train/).

When models are trained, they are bundled with their configuration as follows:

```python
{
    'model_state_dict': model.state_dict(),
    'config': config
}
```

When models are loaded, any manually provided configuration overrides the values bundled with the model.

## Licenses

Most of the code in this repository is under the MIT license. The files rPPG/utils/TSCAN.py and rPPG/utils/PhysFormer.py, which are adapted from the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), are under the Responsible Artificial Intelligence Source Code License contained in the file LICENSE-rPPG-Toolbox.
