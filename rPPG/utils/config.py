#!/usr/bin/env python3

class Config(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)
        for toplevel in ['model', 'training', 'evaluation']:
            if toplevel not in self:
                self[toplevel] = {}

    ##### model #####

    def architecture(self) -> str:
        # Supported: 'CNN3D', 'NRNet', 'NRNet_simple', 'Flex', 'TS-CAN', 'PhysFormer', 'litenetv2', 'litenetv6'
        return self['model'].get('architecture', 'CNN3D')

    def depth(self) -> int:
        # Supported for Flex and TS-CAN models
        return self['model'].get('depth', 10)

    def padding_mode(self) -> str:
        # Supports any padding_mode supported by torch.nn.Conv3d
        # Applies to Flex only
        return self['model'].get('padding_mode', 'replicate')

    def hidden_channels(self) -> int:
        # Hidden layer channels
        return self['model'].get('hidden_channels', 64)

    def flex_dilation(self) -> bool:
        # Perform dilation in flex models
        return self['model'].get('flex_dilation', True)

    def tk(self) -> int:
        return self['model'].get('tk', 5)

    def channels(self) -> str:
        # Supported: Any combination of characters from 'n', 'rgb', 'hsv', 'yuv'
        return self['model'].get('channels', 'rgb')

    def out_channels(self) -> int:
        # Supports values >= 1.
        # If 1, then output is shape [B,T].
        # If >1, then output is shape [B,T,C].
        return self['model'].get('out_channels', 1)

    def num_waves(self) -> int:
        # Used in NRNet_simple
        return self['model'].get('num_waves', 4)

    def frame_width(self) -> int:
        return self['model'].get('frame_width', 64)
    
    def frame_height(self) -> int:
        return self['model'].get('frame_height', 64)

    def fpc(self) -> int:
        # If set to 0 then uses full video length (may result in errors if batch_size > 1)
        # See val_test_fpc
        return self['model'].get('fpc', 136)
    
    def fps(self) -> int:
        return self['model'].get('fps', 30)

    ##### training #####

    def num_workers(self) -> int:
        return self['training'].get('num_workers', 4)
    
    def num_epochs(self) -> int:
        return self['training'].get('num_epochs', 40)

    def augmentation(self) -> str:
        # Supported: Any combination of characters from 'figcsmn'
        return self['training'].get('augmentation', 'figcsmn')

    def negative_probability(self) -> float:
        return self['training'].get('negative_probability', 0.5)

    def noise_width(self) -> float:
        return self['training'].get('noise_width', 3.0)

    def normalization(self) -> str:
        # Supported: 'scale', 'stddev', 'histogram'
        return self['training'].get('normalization', 'scale')

    def dropout(self) -> float:
        return self['training'].get('dropout', 0.5)

    def batch_size(self) -> int:
        return self['training'].get('batch_size', 4)

    def lr(self) -> float:
        return self['training'].get('lr', 0.0001)

    def masks(self) -> bool:
        return self['training'].get('masks', True)

    def loss(self) -> dict:
        # Supported: Dict where keys are:
        #   'negpearson', 'mae', 'mse', 'mcc', 'envelope', 'bandwidth', 'sparsity', 'variance';
        # and values are relative weight (floats)
        # Also accepts keys from negativeLoss, but who would want to do that?
        return self['training'].get('loss', {'negpearson': 1})
    
    def negativeLoss(self) -> dict:
        # Supported: Dict where keys are:
        #   'deviation', 'specentropy', 'specflatness';
        # and values are relative weight (floats)
        # Also accepts keys from loss, but who would want to do that?
        return self['training'].get('negativeLoss', {'deviation': 1})

    ##### evaluation #####

    def val_test_fpc(self) -> int:
        # Defines a fpc for validation and test sets (default: use same fpc as training)
        # Since batch_size always = 1 for val and test, val_test_fpc may freely be set to 0.
        return self['evaluation'].get('val_test_fpc', self.fpc())

    def hr_method(self) -> str:
        # Supported: 'fft', 'peaks', 'cwt', 'sssp'
        return self['evaluation'].get('hr_method', 'fft')

    def hz_low(self) -> float:
        return self['evaluation'].get('hz_low', 0.66666)

    def hz_high(self) -> float:
        return self['evaluation'].get('hz_high', 3.0)

    def fft_window(self) -> float:
        return self['evaluation'].get('fft_window', 10.0)

    def skip_fft(self) -> bool:
        return self['evaluation'].get('skip_fft', True)

    def move_cost(self) -> float:
        return self['evaluation'].get('move_cost', 10.0)

    def smooth_method(self) -> str:
        # Supported: 'none', 'delta_limit', 'smooth'
        return self['evaluation'].get('smooth_method', 'none')

    def delta_limit(self) -> float:
        return self['evaluation'].get('delta_limit', 6.0)

    def smooth(self) -> float:
        return self['evaluation'].get('smooth', 5.0)

    def multiprocessing(self) -> bool:
        return self['evaluation'].get('multiprocessing', False)

    def columns(self) -> list:
        # List of columns when printing evaluation metrics, default prints all
        return self['evaluation'].get('columns', [])
