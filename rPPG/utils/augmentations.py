import numpy as np
from scipy import signal
import torch
import cv2

def createNegativeSample(clip: np.ndarray, noiseWidth: float) -> (np.ndarray, np.ndarray, np.ndarray):
    ''' Creates a negative sample from frame shuffling, uniform noise, or Gaussian noise
    Arguments:
        clip: numpy array of shape [T,H,W,C]
        length: number of time points
    '''
    noiseType = np.random.randint(0,3)
    shape = clip.shape
    T,H,W,C = shape
    if noiseType == 0: #Shuffle
        shuffleIdcs = np.random.permutation(T)
        clip = clip[shuffleIdcs] # shuffle the frame order
    elif noiseType == 1: #Normal
        frameIdx = np.random.randint(T)
        clip[:] = clip[frameIdx] # constant frame throughout clip
        noise = np.random.normal(0, noiseWidth, shape)
        clip = clip + noise
    else: #Uniform
        ## Negative sample via single frame plus dynamic noise
        frameIdx = np.random.randint(T)
        clip[:] = clip[frameIdx] # constant frame throughout clip
        noise = np.random.uniform(-noiseWidth, noiseWidth, shape)
        clip = clip + noise
    HR = np.zeros(T)
    wave = np.zeros(T)
    return clip, wave, HR

def interpolateLinear(clip: np.ndarray, length: int, channelOrder: str) -> np.ndarray:
    '''
    Arguments:
        clip: numpy array of shape [T,H,W,C]
        length: number of time points in output interpolated sequence
        channelOrder: The order of channels as input to training, e.g., "rgb"
    Returns:
        numpy array of shape [C,T,H,W]
    '''
    clip = transformClip(clip, channelOrder)
    clip = clip[np.newaxis,:]
    shape = clip.shape
    clip = torch.from_numpy(clip)
    if len(shape) == 5: # (BxCxTxHxW)
        clip = torch.nn.functional.interpolate(clip, (length, shape[-2], shape[-1]), mode='trilinear', align_corners=False)
    elif len(shape) == 4: # (Bx1xCxT)
        clip = torch.nn.functional.interpolate(clip, (shape[-2], length), mode='bilinear', align_corners=False)
    else:
        print(f'ERROR: Cannot interpolate clip of shape {shape}')
        exit(1)
    return clip.numpy()[0]

def transformClip(clip: np.ndarray, channelOrder: str) -> np.ndarray:
    """ Apply basic clip transformations
    
    Transformations consist of setting the channel order and the shape of the clip 

    Arguments:
      clip: numpy array of shape TxHxWxC
      channelOrder: The order of channels as input to training, e.g., "rgb"

    Returns:
      np.ndarray: The transformed clip of shape CxTxHxW
    """
    if channelOrder:
        channelOrder = channelOrder.lower()
        # We support N, RGB, HSV, YUV.
        # If a V is given we infer based on the presence of H,S,Y,U, defaulting to HSV's V.
        # Arrange the channels: Orig is BGR[N]
        d = {'b':0, 'g':1, 'r':2, 'n':3}
        # Ensure we have a BGR clip to work with:
        BGR = clip[...,0:3]
        def updateClip(channels, conversion):
            cvted = np.array([cv2.cvtColor(frame, conversion) for frame in BGR.astype('float32')])
            for i, c in enumerate(channels):
                d[c] = clip.shape[-1]+i
            return np.concatenate((clip, cvted), axis=len(clip.shape)-1)
        if any(c in channelOrder for c in 'yu'): # Convert to YUV
            clip = updateClip('yuv', cv2.COLOR_BGR2YUV)
        if any(c in channelOrder for c in 'hsv'): # Convert to HSV
            clip = updateClip('hsv', cv2.COLOR_BGR2HSV)
        channelOrder = [d[c] for c in channelOrder]
        clip = clip[...,channelOrder]
    if len(clip.shape) == 4:
        # Clip must be in order (CxTxHxW)
        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float64)
    elif len(clip.shape) == 2:
        # Clip must be in order (1xCxT)
        clip = np.transpose(clip, (1, 0)).astype(np.float64)
        clip = clip[np.newaxis,:,:]
    return clip

def interpolateHW(clip: np.ndarray, h: int, w: int) -> np.ndarray:
    ''' Interpolate the clip height and width
    Arguments:
      clip: numpy array of shape [C,T,H,W]
      h, w: Height and width dimensions

    Returns:
      np.ndarray: The transformed clip of shape CxTxHxW
    '''
    clip = clip[np.newaxis,:] # (BxCxTxHxW)
    clip = torch.from_numpy(clip)
    shape = clip.shape
    clip = torch.nn.functional.interpolate(clip, (shape[-3], h, w), mode='trilinear', align_corners=False)
    return clip.numpy()[0]

def cropRandom(clip, crop_scale_lims=[0.5, 1]):
    ''' Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    '''
    C,T,H,W = clip.shape
    scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    px = np.round(scale * H).astype(int)
    x = np.random.randint(0, H-px+1)
    y = np.random.randint(0, H-px+1)
    return interpolateHW(clip[:,:,y:y+px,x:x+px].copy(), H, W)

def distort(clip: np.ndarray, factor: float) -> np.ndarray:
    ''' Apply a temporal distortion, speeding (or slowing) the clip

    Both the start and end of the clip are affected such that all of the data is used.

    Arguments:
      clip: numpy array of shape [C,T,H,W] or [T] (if it is the wave)
      factor: The factor by which to speed up (factor > 1) or slow down (factor < 1)

    Returns:
      np.ndarray: The distorted clip
    '''
    # 1st try: Use scipy interpolate and hope it works
    from scipy import interpolate
    axis = 0 if len(clip.shape) == 1 else 1
    clipLen = clip.shape[axis]
    f = interpolate.interp1d(range(clipLen), clip, axis=axis, fill_value='extrapolate')
    start = 2/(1+factor)
    end = start*factor
    t = [x*start+x**2*(end-start)/(2*clipLen) for x in range(clipLen)]
    return f(t)

def normalize(clip: np.ndarray, normalization: str = 'scale'):
    # Fix out of bounds caused by illumination or gaussian noise
    clip = np.clip(clip, 0, 255)
    if normalization == 'scale':
        clip = clip / 255 # Scale to [0,1]
    elif normalization == 'stddev':
        clip = clip - clip.min()
        if clip.max() != clip.min():
            #clip = clip / (clip.max()-clip.min())
            clip = clip / clip.std()
    elif normalization == 'histogram':
        # https://stackoverflow.com/a/28520445
        for channel in range(clip.shape[-1]):
            channelArry = clip[...,channel]
            channelShape = channelArry.shape
            hist, bins = np.histogram(channelArry.flatten(), 256, density=True)
            cdf = hist.cumsum()
            cdf = 255 * cdf / cdf[-1]
            clip[...,channel] = np.interp(channelArry.flatten(), bins[:-1], cdf).reshape(channelShape)
    else:
        raise ValueError(f'Unknown normalization technique {normalization}')
    return clip

def applyAugmentations(video: np.ndarray, wave: np.ndarray, HR: np.ndarray, interval: tuple, config: dict) -> (np.ndarray, dict):
    """Apply augmentations to video clip

    Arguments:
      video: The input video
      wave: The ground truth wave (may be None if s not in augmentations)
      HR: The ground truth HR (may be None if s not in augmentations)
      interval: The (start, end) interval for the clip
      config: The config for this run

    Returns:
      np.ndarray: The augmented clip
      dict: Extras, including any of:
       - "wave": the augmented wave
       - "live": the liveness of the sample
       - "speed": the speed factor
       - "mod": the modulation factor
    """
    augs = config.augmentation()
    extras = {'live': 1, 'speed': 1, 'mod': 1}

    # If negative is sampled, we can skip the speed augmentations
    if ('n' in augs) and (np.random.rand() < config.negative_probability()):
        clip, wave, HR = createNegativeSample(video[interval[0]:interval[1]], config.noise_width())
        clip = transformClip(clip, config.channels())
        extras['live'] = 0
    else:
        # Handle speed first
        if 's' in augs:
            def getInterval(arry, intvl, targetLen=None, offset=0) -> np.ndarray:
                if not targetLen:
                    targetLen = intvl[1]-intvl[0]
                center = min(len(arry)-targetLen/2, max(targetLen/2, sum(intvl)/2+offset))
                return (max(0, int(center-targetLen/2)), min(len(arry), int(center+targetLen/2)))
            hrInt = getInterval(HR, interval, offset=(len(HR)-len(wave))/2)
            #print(f'Got interval: {hrInt}')
            avgHR = np.mean(HR[hrInt[0]:hrInt[1]])
            rangeHR = [config.hz_low() * 60, config.hz_high() * 60]
            targetHR = np.random.rand() * (rangeHR[1]-rangeHR[0]) + rangeHR[0]
            factor = targetHR / avgHR
            fpc = (interval[1]-interval[0])
            targetLen = int(fpc * factor)
            # Grab a chunk of video that's (close to) targetLen long, and make it fpc long
            target = getInterval(wave, interval, targetLen=targetLen)
            extras['speed'] = (target[1]-target[0])/fpc
            clip = interpolateLinear(video[target[0]:target[1]], fpc, config.channels())
            wave = signal.resample(wave[target[0]:target[1]], fpc)
        else:
            clip = transformClip(video[interval[0]:interval[1]], config.channels())
            if wave is None:
                wave = np.array([])
            else:
                wave = wave[interval[0]:interval[1]]
        if 'm' in augs:
            # Modulate
            # Max change is 7 bpm/sec
            fpc = (interval[1]-interval[0])
            seconds = fpc / config.fps()
            avgHR = 100 if HR is None else np.mean(HR)
            # (avgHR * factorMax - avgHR) / seconds = 7
            # (seconds * 7 + avgHR) / avgHR - 1 = factorMax
            factor = np.random.rand() * ((seconds * 7 + avgHR) / avgHR - 1) + 1
            if np.random.rand() > 0.5:
                factor = 1/factor
            extras['mod'] = factor
            clip = distort(clip, factor)
            wave = distort(wave, factor)

    # All other augmentations occur after transformClip (only valid for videos)
    if len(clip.shape) == 4:
        h, w = config.frame_height(), config.frame_width()
        if h != clip.shape[-2] or w != clip.shape[-1]:
            clip = interpolateHW(clip, h, w)
        if 'f' in augs:
            if np.random.rand() > 0.5:
                clip = np.flip(clip, 3)
            # Also rotate
            #clip = np.rot90(clip, k=np.random.randint(0, 4), axes=(-2, -1))
        if 'i' in augs:
            clip += np.random.normal(0, 10)
        if 'g' in augs:
            clip += np.random.normal(0, 2, clip.shape)
        if 'c' in augs:
            clip = cropRandom(clip)
        clip = normalize(clip, config.normalization())
    if not (wave is None or len(wave) == 0):
        # Also scale down wave (only affects amplitude-sensitive losses like MAE)
        scaleFrom = [80, 120]
        scaleTo = [-.5, .5]
        scaleFactor = (scaleTo[1]-scaleTo[0]) / (scaleFrom[1]-scaleFrom[0])
        wave = wave * scaleFactor - scaleFrom[0]*scaleFactor + scaleTo[0]
        extras['wave'] = wave
    return clip, extras
