import math

import numpy as np

import pywt
import scipy.signal as signal
from pyentrp import entropy as ent
from scipy import stats
from scipy.stats import norm


def psd(x, start_freq, end_freq, fs=1000.0):
    """
    freq resolution (interval of f ): sampling freq / number of sample points
    impossible to measure freq above the Nyquist limit(Fs/2),
    get rid of above Nyquist limit signal, and double the magnitude by 2
    to make the total amplitude equal to one, need to normalize by number of sample

    Args:
        :param x: 1-dim list
        :param start_freq: the start frequency of psd calculation (float)
        :param end_freq: the end frequency of psd calculation (float)
        :param fs: the sampling frequency (float)
    Return:
        the power spectral density of signal x
    """
    # f: array of sample frequencies, Pxx: PSD of X
    f, Pxx = signal.periodogram(x, fs)  # Estimate power spectral density

    Pxx = Pxx / (np.sum(Pxx) + 1e-6)

    start_idx = np.argmin(np.abs(f - start_freq))
    end_idx = np.argmin(np.abs(f - end_freq))
    psdensity = np.mean((Pxx[start_idx:end_idx])**2)

    return psdensity


def spectral_entropy(x, start_freq, end_freq, fs=1000.0):
    """Calculate the spectral entropy of signal x
    NOTE: https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy

    Args:
        :param x: 1-dim list
        :param start_freq: the start frequency of psd calculation (float)
        :param end_freq: the end frequency of psd calculation (float)
        :param fs: the sampling frequency (float)
    Return:
        the spectral entropy of signal x
    """
    f, Pxx = signal.periodogram(x, fs)
    Pxx = np.square(Pxx) / len(Pxx)
    Pxx = Pxx / (np.sum(Pxx) + 1e-6)

    start_idx = np.argmin(np.abs(f - start_freq))
    end_idx = np.argmin(np.abs(f - end_freq))
    p = Pxx[start_idx:end_idx]

    return np.sum(p*np.log(p + 1e-6)) * -1


def STFT_se(x):
    """Calculate the mean spectral entropy of signal over time on x
    NOTE: https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy

    Args:
        :param x: 1-dim list
    Return:
        the mean spectral entropy of signal over time of signal x
    """

    f, t, Zxx = signal.stft(x, fs=1000.0, nperseg=125, nfft=5000)  # nperseg: time resolution, nfft: freq resolution
    Zxx = Zxx / np.sum(Zxx)  # Zxx: 2501 x 160, f: 2501, t: 160

    Zxx = np.transpose(Zxx)  # 160 x 2501
    se_list = []
    for t in range(len(Zxx)):
        freq = Zxx[t]
        Ej = np.abs(freq)
        E = np.sum(Ej)
        p = Ej / E
        se = np.sum(p*np.log(p+1e-6)) * -1
        se_list.append(se)

    return np.mean(se_list)


def mean_power(x):
    """Return the mean power of given signal

    Args:
        :param x: 1-dim list
    Return:
        the mean power of x
    """
    return np.mean(np.square(x))


def mean(x):
    """Return the mean absolute value of given signal

    Args:
        :param x: 1-dim list
    Return:
        the mean absolute value of x
    """
    return np.mean(np.abs(x))


def std(x):
    """Return the standard deviation of given signal

    Args:
        :param x: 1-dim list
    Return:
        the standard deviation of x
    """
    return np.std(x)


def first_diff(x):
    """Return the mean value of first derivative of given signal

    Args:
        :param x: 1-dim list
    Return:
        the the mean value of first derivative of x
    """
    return np.mean(np.abs(np.diff(x)))


def hjcomp(x):
    """Return the Hjorth complexity of given signal
    # NOTE: https://en.wikipedia.org/wiki/Hjorth_parameters

    Args:
        : param x: 1-dim list
    Return:
        the the Hjorth complexity of x
    """

    def _mob(x):
        return np.sqrt(np.var(np.diff(x))/np.var(x))
    return _mob(np.diff(x))/_mob(x)


"""
def second_diff(x):
    order = 2
    if(order == 0):
        array = x
    else:
        array = x[order:] - x[0:-order]
    return np.mean(np.abs(array))
"""


def wl_mean_power(x):
    """Return the mean power of each sub-band ('cA7', 'cD7', 'cD6', 'cD5', 'cD4') using wavelet transform

        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------

        If the sampling rate is 1000
        level = 7
        cD1: 250-500
        cD2: 125-250
        cD3: 62.5-125
        cD4: 31.25-62.5
        cD5: 15.625-31.25
        cD6: 7.81-15.625  alpha
        cD7: 3.9-7.8  theta
        cA7: 0-3.9

    Args:
        :param x: 1-dim list
    Return:
        the mean power of x in each sub-band ('cA7', 'cD7', 'cD6', 'cD5', 'cD4') (returned type: list of float)
    """
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    feature = [np.mean(np.square(coeff)) for coeff in list_coeffs]

    return np.array(feature)


def wl_mean(x):
    """Return the mean abs value of each sub-band ('cA7', 'cD7', 'cD6', 'cD5', 'cD4') using wavelet transform

    Args:
        :param x: 1-dim list
    Return:
        the mean abs value of x in each sub-band (returned type: list of float)
    """
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    feature = [np.mean(np.abs(coeff)) for coeff in list_coeffs]

    return np.array(feature)


def wl_std(x):
    """Return the standard deviation of each sub-band ('cA7', 'cD7', 'cD6', 'cD5', 'cD4') using wavelet transform

    Args:
        :param x: 1-dim list
    Return:
        the standard deviation of x in each sub-band (returned type: list of float)
    """
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    feature = [np.std(coeff) for coeff in list_coeffs]

    return np.array(feature)


def wl_ratio_of_subbands(x):
    """Return the ratio of the abs mean values of adjacent sub-bands

    Args:
        :param x: 1-dim list
    Return:
        the ratio of the abs mean values of adjacent sub-bands (returned type: list of float)

        ex: the first index of the return is the ratio of cA7 and cD7
            the second index of the return is the ratio of cD7 and cD6
    """
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    feature = [np.mean(np.abs(list_coeffs[i])) / np.mean(np.abs(list_coeffs[i + 1])) for i in range(len(list_coeffs)-1)]

    return np.array(feature)


def wl_entropy(x):
    """Return the entropy of each sub-band ('cA7', 'cD7', 'cD6', 'cD5', 'cD4') using wavelet transform

    Args:
        :param x: 1-dim list
    Return:
        the entropy of x in each sub-band (returned type: list of float)
    """
    def _calculate_entropy(x):
        """calculate the entropy of given signal

        Args:
            :param x: 1-dim list
        Return:
            the entropy of x
        """
        bin_len = int(len(x)/5) if int(len(x)/5) > 0 else 2
        hist, bin_edges = np.histogram(x, bins=bin_len)
        probabilities = [elem/len(x) for elem in hist]
        entropy = stats.entropy(probabilities)
        return entropy

    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    feature = [_calculate_entropy(coeff) for coeff in list_coeffs]

    return np.array(feature)


def wl_spec_entropy(x):
    """Return the spectral entropy of each sub-band ('cA7', 'cD7', 'cD6', 'cD5', 'cD4') using wavelet transform

    Args:
        :param x: 1-dim list
    Return:
        the entropy of x in each sub-band (returned type: list of float)
    """
    def _spec_entropy(x):
        """calculate the spectral entropy of given signal

        Args:
            :param x: 1-dim list
        Return:
            the spectral entropy of x
        """
        Ej = np.abs(x)
        E = np.sum(Ej)
        p = Ej / E
        return np.sum(p*np.log(p+1e-6)) * -1

    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    feature = [_spec_entropy(coeff) for coeff in list_coeffs]

    return np.array(feature)


def multiscale_permutation_entropy(x, m=3, delay=1, scale=20):
    """Return multiscale permutation entropy of the given signal

    Args:
        :param x: 1-dim list
    Return:
        the multiscale permutation entropy of x (returned type: list of float)
    """
    return ent.multiscale_permutation_entropy(x, m=m, delay=delay, scale=scale)


def multiscale_entropy(x, sample_length=2, maxscale=20):
    """Return multiscale entropy of the given signal

    Args:
        :param x: 1-dim list
    Return:
        the multiscale entropy of x (returned type: list of float)
    """
    return ent.multiscale_entropy(x, sample_length=sample_length, maxscale=maxscale)


def coarse_graining(org_signal, scale):
    """Coarse-graining the signals.

    Args:
        org_signal: original signal,
        scale: desired scale
    Return:
        new_signal: coarse-grained signal
    """
    new_length = int(np.fix(len(org_signal) / scale))
    new_signal = np.zeros(new_length)
    for i in range(new_length):
        new_signal[i] = np.mean(org_signal[i * scale:(i + 1) * scale])

    return new_signal


def cal_dispersion_entropy(x, d, m, c):
    """Calcuate the dispersion entropy of given signal
    # NOTE: https://ieeexplore.ieee.org/document/7434608

    Args:
        :param x: signals (1-dim list or array)
        :param d: delay
        :param m: embedding dimension
        :param c: the number of classes
    Return:
        the dispersion entropy of x
    """
    y = norm.cdf(x, loc=np.mean(x), scale=np.std(x))
    z = np.round(c*y+0.5)

    num_disp_pattern = len(x) - (m - 1) * d
    pattern_set = {}
    for i in range(num_disp_pattern):
        pattern = ','.join([str(int(z[i:i+m][j])) for j in range(len(z[i:i+m]))])
        if pattern in pattern_set:
            pattern_set[pattern] += 1
        else:
            pattern_set[pattern] = 1

    disp_entropy = 0
    for key, value in pattern_set.items():
        prob = value / num_disp_pattern
        disp_entropy -= (prob) * math.log(prob)

    return disp_entropy


def multiscale_dispersion_entropy(signal, maxscale=20, classes=6, emb_dim=3, delay=1):
    """Calculate multiscale dispersion entropy

    Args:
        :param signal: input signal,
        :param scale: coarse graining scale,
        :param classes: number of classes,
        :param emd_dim: embedding dimension,
        :param delay: time delay
    Return:
        multiscale dispersion entropy value of the signal
    """
    mde = np.zeros(maxscale)
    for i in range(maxscale):
        cg_signal = coarse_graining(signal, i+1)
        en = cal_dispersion_entropy(cg_signal, d=delay, m=emb_dim, c=classes)
        mde[i] = en

    return mde


def cal_fluctuation_dispersion_entropy(x, d, m, c):
    """Calcuate the dispersion entropy of given signal
    # NOTE: https://arxiv.org/pdf/1902.10825.pdf

    Args:
        :param x: signals (1-dim list or array)
        :param d: delay
        :param m: embedding dimension
        :param c: the number of classes
    Return:
        the dispersion entropy of x
    """
    y = norm.cdf(x, loc=np.mean(x), scale=np.std(x))
    z = np.round(c*y+0.5)

    num_dispersion_pattern = len(x) - (m - 1) * d
    pattern_set = {}
    for i in range(num_dispersion_pattern):
        pattern = ','.join([str(int(z[i:i+m][j]-np.min(z[i:i+m]))) for j in range(len(z[i:i+m]))])

        if pattern in pattern_set:
            pattern_set[pattern] += 1
        else:
            pattern_set[pattern] = 1

    fluctuation_dispersion_entropy = 0
    for key, value in pattern_set.items():
        prob = value / num_dispersion_pattern
        fluctuation_dispersion_entropy -= (prob) * math.log(prob)

    return fluctuation_dispersion_entropy


def multiscale_fluctuation_based_dispersion_entropy(signal, maxscale=20, classes=6, emb_dim=3, delay=1):
    """ Calculate multiscale fluctuation_based dispersion entropy.
    # NOTE: https://arxiv.org/pdf/1902.10825.pdf

    Args:
        :param signal: input signal,
        :param scale: coarse graining scale,
        :param classes: number of classes,
        :param emd_dim: embedding dimension,
        :param delay: time delay
    Return:
        mde: multiscale dispersion entropy value of the signal (list of float)
    """
    mfde = np.zeros(maxscale)
    for i in range(maxscale):
        cg_signal = coarse_graining(signal, i+1)
        en = cal_fluctuation_dispersion_entropy(cg_signal, d=delay, m=emb_dim, c=classes)
        mfde[i] = en

    return mfde


def wl_mpe(x):
    """Return multiscale permutation entropy of each sub-band

    Args:
        :param x: 1-dim list
    Return:
        multiscale permutation entropy of each sub-band (returned type: list of float)
    """
    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    for i, coeff in enumerate(list_coeffs):
        if i == 0 or i == 1:
            c = 2
            m = 2
        elif i == 1:
            c = 3
            m = 2
        else:
            c = 3
            m = 3

        entropy = multiscale_permutation_entropy(coeff, m=m, delay=1, scale=20)
        feature = feature + entropy

    return np.array(feature)


def wl_mde(x):
    """Return multiscale dispersion entropy of each sub-band

    Args:
        :param x: 1-dim list
    Return:
        multiscale dispersion entropy of each sub-band (returned type: list of float)
    """

    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    for i, coeff in enumerate(list_coeffs):
        if i == 0 or i == 1:
            c = 2
            m = 2
        elif i == 1:
            m = 2
            c = 3
        else:
            c = 3
            m = 3
        entropy = list(multiscale_dispersion_entropy(coeff, maxscale=20, classes=c, emb_dim=m, delay=1))
        feature = feature + entropy

    return np.array(feature)


def wl_mfde(x):
    """Return multiscale fluctuation-based dispersion entropy of each sub-band

    Args:
        :param x: 1-dim list
    Return:
        multiscale fluctuation-based dispersion entropy of each sub-band (returned type: list of float)
    """

    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    for i, coeff in enumerate(list_coeffs):
        if i == 0 or i == 1:
            c = 2
            m = 2
        elif i == 1:
            m = 2
            c = 3
        else:
            c = 3
            m = 3
        entropy = list(multiscale_fluctuation_based_dispersion_entropy(
            coeff, maxscale=20, classes=c, emb_dim=m, delay=1))
        feature = feature + entropy

    return np.array(feature)
