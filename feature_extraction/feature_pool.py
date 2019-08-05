import scipy.signal as signal
import numpy as np
import pywt
from scipy import stats
from pyentrp import entropy as ent
from scipy.stats import norm
import math


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
    s_freq = 3.0
    e_freq = 30.0
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
    return np.mean(np.square(x))


def mean(x):
    return np.mean(np.abs(x))


def std(x):
    return np.std(x)


def first_diff(x):
    return np.mean(np.abs(np.diff(x)))


def hjcomp(x):
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


def WLMeanPower(x):
    '''
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
        level = 7
        cD1: 250-500
        cD2: 125-250
        cD3: 62.5-125
        cD4: 31.25-62.5
        cD5: 15.625-31.25 
        cD6: 7.81-15.625  alpha
        cD7: 3.9-7.8  theta
        cA7: 0-3.9
    '''
    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    list_coeffs = list_coeffs[0:5]

    for coeff in list_coeffs:
        power = np.mean(np.square(coeff))
        feature.append(power)

    return np.array(feature)


def WLMean(x):
    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    #list_coeffs = list_coeffs[2:5]
    list_coeffs = list_coeffs[0:5]

    for coeff in list_coeffs:
        AP = np.mean(np.abs(coeff))
        feature.append(AP)
    return np.array(feature)


def WLstd(x):
    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    #list_coeffs = list_coeffs[2:5]
    list_coeffs = list_coeffs[0:5]

    for coeff in list_coeffs:
        AP = np.std(coeff)
        feature.append(AP)
    return np.array(feature)


def WLRAM(x):
    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    #list_coeffs = list_coeffs[2:5]
    list_coeffs = list_coeffs[0:5]

    for i in range(len(list_coeffs)-1):
        RAM = np.mean(np.abs(list_coeffs[i])) / np.mean(np.abs(list_coeffs[i + 1]))
        feature.append(RAM)
    return np.array(feature)


def WLEntropy(x):
    def _calculate_entropy(list_values):
        bin_len = int(len(list_values)/5) if int(len(list_values)/5) > 0 else 2
        hist, bin_edges = np.histogram(list_values, bins=bin_len)
        #counter_values = Counter(list_values).most_common()
        #probabilities = [elem[1]/len(list_values) for elem in counter_values]
        probabilities = [elem/len(list_values) for elem in hist]
        entropy = stats.entropy(probabilities)
        return entropy
    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    #list_coeffs = list_coeffs[2:5]
    list_coeffs = list_coeffs[0:5]

    for coeff in list_coeffs:
        entropy = _calculate_entropy(coeff)
        feature.append(entropy)
    return np.array(feature)


def WLSpecralEntropy(x):
    def _spec_entropy(coeff):
        Ej = np.abs(coeff)
        E = np.sum(Ej)
        p = Ej / E
        return np.sum(p*np.log(p+1e-6)) * -1

    feature = []
    list_coeffs = pywt.wavedec(x, 'sym10', level=7)
    #print([len(coeff) for coeff in list_coeffs])
    list_coeffs = list_coeffs[0:5]

    for coeff in list_coeffs:
        entropy = _spec_entropy(coeff)
        feature.append(entropy)
    return np.array(feature)


def multiscale_permutation_entropy(x, m=3, delay=1, scale=20):
    return ent.multiscale_permutation_entropy(x, m=m, delay=delay, scale=scale)


def multiscale_entropy(x, sample_length=2, maxscale=20):
    return ent.multiscale_entropy(x, sample_length=sample_length, maxscale=maxscale)


def coarse_graining(signal, scale):
    """Coarse-graining the signals.
    Arguments:
        signal: original signal,
        scale: desired scale
    Return:
        new_signal: coarse-grained signal
    """
    new_length = int(np.fix(len(signal) / scale))
    new_signal = np.zeros(new_length)
    for i in range(new_length):
        new_signal[i] = np.mean(signal[i * scale:(i + 1) * scale])

    return new_signal


def DispEn(x, d, m, c):
    '''
        d: delay
        m: embedding dimension
        c: number of class
    '''
    y = norm.cdf(x, loc=np.mean(x), scale=np.std(x))
    z = np.round(c*y+0.5)
    num_DispPattern = len(x) - (m - 1) * d
    pattern_set = {}
    for i in range(num_DispPattern):
        pattern = ','.join([str(int(z[i:i+m][j])) for j in range(len(z[i:i+m]))])
        if pattern in pattern_set:
            pattern_set[pattern] += 1
        else:
            pattern_set[pattern] = 1

    dispEn = 0
    for key, value in pattern_set.items():
        prob = value / num_DispPattern
        dispEn -= (prob) * math.log(prob)

    return dispEn


def multiscale_dispersion_entropy(signal, maxscale=20, classes=6, emb_dim=3, delay=1):
    """ Calculate multiscale dispersion entropy.
    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mde: multiscale dispersion entropy value of the signal
    """
    mde = np.zeros(maxscale)
    for i in range(maxscale):
        cg_signal = coarse_graining(signal, i+1)
        en = DispEn(cg_signal, d=delay, m=emb_dim, c=classes)
        mde[i] = en

        #cg_signal = coarse_graining(signal, scale)
        #prob = dispersion_frequency(cg_signal, classes, emb_dim, delay)
        #prob = list(filter(lambda p: p != 0., prob))
        #mde = -1 * np.sum(prob * np.log(prob))

    return mde


def FDispEn(x, d, m, c):
    '''
        d: delay
        m: embedding dimension
        c: number of class
    '''
    y = norm.cdf(x, loc=np.mean(x), scale=np.std(x))
    z = np.round(c*y+0.5)

    num_DispPattern = len(x) - (m - 1) * d
    pattern_set = {}
    for i in range(num_DispPattern):
        pattern = ','.join([str(int(z[i:i+m][j]-np.min(z[i:i+m]))) for j in range(len(z[i:i+m]))])

        if pattern in pattern_set:
            pattern_set[pattern] += 1
        else:
            pattern_set[pattern] = 1

    dispEn = 0
    for key, value in pattern_set.items():
        prob = value / num_DispPattern
        dispEn -= (prob) * math.log(prob)

    return dispEn


def multiscale_fluctuation_based_dispersion_entropy(signal, maxscale=20, classes=6, emb_dim=3, delay=1):
    """ Calculate multiscale dispersion entropy.
    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mde: multiscale dispersion entropy value of the signal
    """
    mfde = np.zeros(maxscale)
    for i in range(maxscale):
        cg_signal = coarse_graining(signal, i+1)
        en = FDispEn(cg_signal, d=delay, m=emb_dim, c=classes)
        mfde[i] = en

    return mfde


def WLMPE(x):
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
        entropy = multiscale_permutation_entropy(coeff, m=m, delay=1, scale=20)
        feature = feature + entropy
    return np.array(feature)


def WLMDE(x):
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


def WLMFDE(x):
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
