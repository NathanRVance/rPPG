import numpy as np
import scipy.signal


def bandpass_butter(sig, fps, order=4, cutoffs=[42/60, 240/60], btype='bandpass'):
    b, a = scipy.signal.butter(order, Wn=cutoffs, btype=btype, fs=fps)
    res = scipy.signal.filtfilt(b, a, sig, axis=0)
    return res


def temporal_normalization(sig):
    mean_color = np.mean(sig, axis=0)
    if np.array_equal(mean_color, [0,0,0]):
        normed_sig = sig
    else:
        normed_sig = (sig / mean_color) - 1
    return normed_sig



def process_CHROM(signals, framerate, windowed=False):
    '''
        Runs the Chrominance (CHROM) algorithm from a spatially-averaged BGR trace.
        For more details see: De Haan and Jeanne. Robust Pulse Rate from Chrominance-Based rPPG. 2013.

        Arguments
            signals: [T,3] np.array where each time point is an [B,G,R] vector.
            framerate: sampling rate of video which was spatially averaged.
            windowed: whether to glue together small segments during estimation.
        Returns
            H: [T] np.array of the estimated waveform.
    '''

    CHROM = np.array([
        [3, -2, 0],
        [1.5, 1, -1.5]
    ])

    # CHROM matrix expects RGB order
    signals = signals[:,[2,1,0]]

    if windowed:
        N = signals.shape[0]
        pulse = np.zeros(N)
        l = int(framerate * 1.6)
        l = 32 if l < 32 else l
        hanning = np.hanning(l)
        for t in range(0, N-l, l//2):
            sig_win = signals[t:t+l]
            norm_sig = temporal_normalization(sig_win)
            x, x = np.matmul(CHROM, norm_sig.T) #2xT
            x_bandpassed = bandpass_butter(Xs, framerate)
            y_bandpassed = bandpass_butter(Ys, framerate)
            X_std = np.std(x_bandpassed)
            Y_std = np.std(y_bandpassed)
            Y_std = 0.0001 if math.isclose(Y_std,0) else Y_std
            alpha = X_std / Y_std
            S = x_bandpassed - alpha*y_bandpassed
            S_std = np.std(S)
            S_std = 0.0001 if math.isclose(S_std,0) else S_std
            S = (S - np.mean(S)) / S_std
            pulse[t:t+l] = pulse[t:t+l] + hanning*S

    else:
        # Temporally normalize traces
        signals = temporal_normalization(signals)

        # Project RGB signals onto CHROM plane
        x, y = np.matmul(CHROM, signals.T) #2xT

        # now that we have the chrominance signals, apply bandpass
        x_bandpassed = bandpass_butter(x, framerate)
        y_bandpassed = bandpass_butter(y, framerate)

        # build the final pulse signal
        alpha = np.std(x_bandpassed) / np.std(y_bandpassed)
        pulse = x_bandpassed - alpha * y_bandpassed

    return -pulse


