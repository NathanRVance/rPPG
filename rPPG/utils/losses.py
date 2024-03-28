import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
import inspect
import functools

EPSILON=1e-6

def negpearson(x, y):
    ## epsilon is used in denominator to ensure numerical stability
    if len(x.shape) < 2:
        x = torch.reshape(x, (1,-1))
    mean_x = torch.mean(x, 1)
    mean_y = torch.mean(y, 1)
    xm = x.sub(mean_x[:, None])
    ym = y.sub(mean_y[:, None])
    r_num = torch.einsum('ij,ij->i', xm, ym)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1) + EPSILON
    r_vals = r_num / r_den
    return 1 - r_vals

def torch_std(x):
    if len(x.shape) < 2:
        x = torch.reshape(x, (1,-1))
    return torch.std(x, dim=1)

def torch_spectral_entropy(x, nfft=5400, fps=90):
    freqs, psd = torch_power_spectral_density(x, nfft, fps)
    N = psd.shape[1]
    H = - torch.sum(psd*torch.log2(psd+EPSILON), dim=1) ## entropy
    H_n =  H / np.log2(N) ## normalized entropy
    return 1 - H_n

def torch_spectral_flatness(x, nfft=5400, fps=90):
    freqs, psd = torch_power_spectral_density(x, nfft, fps)
    N = psd.shape[1]
    norm = 1 / N
    geo = torch.exp(norm * torch.sum(torch.log(psd+EPSILON), dim=1))
    arith = torch.mean(psd, dim=1)
    sf = geo / arith
    return 1 - sf

def applyBandpass(freqs, psd, hz_low, hz_high, invert=False) -> (np.ndarray, np.ndarray):
    ''' Zero out psd not in bandpass range (does not mutate input)
    '''
    psd = psd.clone()
    # Enforce hz_low and hz_high same first dimension of x
    def toArry(maybeNot) -> np.ndarray:
        if hasattr(maybeNot, 'shape'):
            return maybeNot.to(psd.get_device())
        return torch.tensor([maybeNot]).expand(psd.shape[0]).to(psd.get_device())
    hz_low, hz_high = (toArry(hz) for hz in (hz_low, hz_high))
    freqsExp = freqs[None,:].expand(psd.shape[0], -1)
    if invert:
        freq_idcs = torch.logical_or(freqsExp >= hz_low[:,None], freqsExp <= hz_high[:,None])
    else:
        freq_idcs = torch.logical_or(freqsExp < hz_low[:,None], freqsExp > hz_high[:,None])
    psd[freq_idcs] = 0
    return psd

def torch_power_spectral_density(x, nfft=5400, fps=90, hz_low=None, hz_high=None):
    centered = x - torch.mean(x, keepdim=True, dim=1)
    psd = torch.abs(torch.fft.rfft(centered, n=nfft, dim=1))**2
    N = psd.shape[1]
    freqs = torch.fft.rfftfreq(2*N-1, 1/fps).to(psd.get_device())
    psd = applyBandpass(freqs, psd, hz_low, hz_high)
    psd = psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities
    return freqs, psd

def ipr_ssl(x, nfft=5400, fps=90, hz_low=None, hz_high=None):
    # bandwidth loss
    freqs, psd = torch_power_spectral_density(x, nfft, fps, 0, float('inf'))
    #import cursor
    #cursor.show()
    #breakpoint()
    use_energy, zero_energy = (torch.sum(applyBandpass(freqs, psd, hz_low, hz_high, invert=invert), dim=1) for invert in (False, True))
    return zero_energy / (use_energy + zero_energy + EPSILON)

def emd_ssl(x, nfft=5400, fps=90, hz_low=None, hz_high=None):
    # variance loss
    freqs, psd = torch_power_spectral_density(x, nfft, fps, hz_low, hz_high)
    B,T = psd.shape
    psd = torch.sum(psd, dim=0) / B
    expected = ((1/T)*torch.ones(T)).to(x.get_device()) #uniform distribution
    return torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))

def snr_ssl(x, nfft=5400, fps=90, hz_low=None, hz_high=None, hz_delta=0.1):
    # sparsity loss
    freqs, psd = torch_power_spectral_density(x, nfft, fps, hz_low, hz_high)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq = freqs[signal_freq_idx].view(-1,1)
    freqs = freqs.repeat(psd.shape[0],1)
    band_idcs = torch.logical_and(freqs >= signal_freq-hz_delta, freqs <= signal_freq+hz_delta).to(x.get_device())
    signal_band = torch.sum(psd * band_idcs, dim=1)
    noise_band = torch.sum(psd * torch.logical_not(band_idcs), dim=1)
    return noise_band / (signal_band + noise_band + EPSILON)

def MCC(preds, labels, fps: float = 90, hz_low: float = None, hz_high: float = None):
    # Negative Max Cross Corr
    # See https://github.com/ToyotaResearchInstitute/RemotePPG/blob/master/iccv/src/losses/NegativeMaxCrossCorr.py
    # Normalize
    preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
    labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

    # Zero-pad signals to prevent circular cross-correlation
    # Also allows for signals of different length
    # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
    min_N = min(preds.shape[-1], labels.shape[-1])
    padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
    preds_pad = torch.nn.functional.pad(preds_norm, (0, padded_N - preds.shape[-1]))
    labels_pad = torch.nn.functional.pad(labels_norm, (0, padded_N - labels.shape[-1]))

    # FFT
    preds_fft = torch.fft.rfft(preds_pad, dim=-1)
    labels_fft = torch.fft.rfft(labels_pad, dim=-1)
    
    # Cross-correlation in frequency space
    X = preds_fft * torch.conj(labels_fft)
    X_real = torch.view_as_real(X)

    # Determine ratio of energy between relevant and non-relevant regions
    use_energy = []
    zero_energy = []
    for b in range(X_real.shape[0]):
        freqs = torch.linspace(0, fps/2, X.shape[-1])
        use_freqs = torch.logical_and(freqs >= hz_low[b], freqs <= hz_high[b])
        zero_freqs = torch.logical_not(use_freqs)
        use_energy.append(torch.sum(torch.linalg.norm(X_real[b,use_freqs], dim=-1), dim=-1))
        zero_energy.append(torch.sum(torch.linalg.norm(X_real[b,zero_freqs], dim=-1), dim=-1))
    use_energy = torch.Tensor(use_energy).to(X_real.get_device())
    zero_energy = torch.Tensor(zero_energy).to(X_real.get_device())
    denom = use_energy + zero_energy
    energy_ratio = torch.ones_like(denom)
    for ii in range(len(denom)):
        if denom[ii] > 0:
            energy_ratio[ii] = use_energy[ii] / denom[ii]

    # Zero out irrelevant freqs
    X[:,zero_freqs] = 0.

    # Inverse FFT and normalization
    cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

    # Max of cross correlation, adjusted for relevant energy
    max_cc = torch.max(cc, dim=-1)[0] / energy_ratio
    
    cov = -max_cc

    denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)
    output = torch.zeros_like(cov)
    for ii in range(len(denom)):
        if denom[ii] > 0:
            output[ii] = cov[ii] / denom[ii]
    return output

def envelope(x, y, fps: float = 90, hz_low=None, hz_high=None, weight=(1/3, 1/3, 1/3)):
    ''' Calculate the pearsonr correlation of the envelope

    Arguments:
      x: The predicted labels
      y: The ground truth labels
      weight: Weights for full, top, and bottom envelopes, respectively
    '''
    from rPPG.utils import envelope
    def envelopeWinmax(signals):
        return envelope.winmax(signals, fps, hz_low, hz_high)
    def envelopePeaks(signals):
        return torch.tensor(np.array([envelope.peaks(signal, fps, hz_low, hz_high, extrapolate=True) for signal in signals])).cuda().float()
    if len(x.shape) == 3:
        topx, botx = [x[:,:,1], x[:,:,2]]
        y = y.cpu() # Do peak detection on the CPU
        topy, boty = [envelopePeaks(y), envelopePeaks(-y)]
    else:
        topy, boty = [envelopeWinmax(y), envelopeWinmax(-y)]
        topx, botx = [envelopeWinmax(x), envelopeWinmax(-x)]
    return negpearson(torch.hstack([topx, botx]), torch.hstack([topy, boty])) * weight[0] + negpearson(topx, topy) * weight[1] + negpearson(botx, boty) * weight[2]

class Loss():
    def __init__(self, config):
        self.config = config

    def loss(self, pred, extras: dict) -> (float, dict):
        ''' Calculate the loss for this prediction

        Arguments:
          pred: The predicted wave
          extras: A dictionary containing the following:
           - "wave": the augmented ground truth wave
           - "live": the liveness of the sample
           - "speed": the speed factor
           - "mod": the modulation factor
        
        Returns:
          float: The (pytorch) loss value
          dict: A dictionary of loss components (for diagnostic purposes)
        '''
        lossParts = {}
        for lossDict, isLive in [(self.config.loss(), True), (self.config.negativeLoss(), False)]:
            if type(lossDict) is str:
                # Backwards compatibility
                lossDict = {lossDict: 1}
            for lossName, weight in lossDict.items():
                if weight == 0:
                    continue
                passOn = {} # Some components are loss function specific
                if lossName == 'negpearson':
                    func = negpearson
                elif lossName == 'mae':
                    l1 = torch.nn.L1Loss(reduction='none')
                    func = lambda x, y: torch.mean(l1(x, y), dim=1)
                elif lossName == 'mse':
                    mse = torch.nn.MSELoss(reduction='none')
                    func = lambda x, y: torch.mean(mse(x, y), dim=1)
                elif lossName == 'mcc':
                    func = MCC
                elif lossName == 'envelope':
                    func = envelope
                    if type(weight) is list:
                        passOn['weight'] = weight
                        weight = 1 # Already applied within function
                elif lossName == 'bandwidth':
                    func = ipr_ssl
                elif lossName == 'sparsity':
                    func = snr_ssl
                elif lossName == 'variance':
                    func = emd_ssl
                elif lossName == 'deviation':
                    func = torch_std
                elif lossName == 'specentropy':
                    func = torch_spectral_entropy
                elif lossName == 'specflatness':
                    func = torch_spectral_flatness
                else:
                    raise ValueError(f'ERROR: Unknown loss function {lossName}')
                predAdjusted = pred
                if lossName != 'envelope' and len(pred.shape) == 3:
                    predAdjusted = pred[:,:,0]
                keywords = [p.name for p in inspect.signature(func).parameters.values()]
                passOn.update({key: value for key, value in [('fps', self.config.fps()), ('hz_low', self.config.hz_low()), ('hz_high', self.config.hz_high())] if key in keywords})
                for key in passOn.keys() & ['hz_low', 'hz_high']:
                    if 'speed' in extras: # Adjust freq bounds based on augmented speed
                        passOn[key] = extras['speed'] * passOn[key]
                    else: # Adjust dimensionality
                        passOn[key] = torch.ones(pred.shape[0]) * passOn[key]

                func = functools.partial(func, **passOn)
                spec = inspect.getfullargspec(func)
                numPositional = len(spec.args) - (0 if not spec.defaults else len(spec.defaults))
                liveness = extras['live'].to(pred.get_device())
                if not isLive:
                    liveness = 1 - liveness
                if numPositional == 1:
                    lossPart = weight * liveness * func(predAdjusted)
                elif numPositional == 2:
                    lossPart = weight * liveness * func(predAdjusted, extras['wave'].to(pred.get_device()))
                else:
                    raise ValueError(f'ERROR: Loss function {lossName} requires {numPositional} arguments, but only 1 or 2 supported!')
                if pred.shape[0] > 1: # Batch size > 1
                    lossPart = torch.mean(lossPart)
                lossParts[lossName] = lossPart
        return torch.sum(torch.stack(list(lossParts.values())), dim=0), lossParts
