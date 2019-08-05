import numpy as np
from tqdm import tqdm
import os
from feature_extraction.utils import freq_domain_extract
import feature_extraction.feature_pool as fp

EPOCH_PATH = '/home/access/cys/SART_Paper/user_epoch_eeg_npz'  # Load the epoch data
FEATURE_PATH = '/home/access/cys/SART_Paper/feature_npz'  # saved feature path

ICA = False  # Use ICA or not
BEFORE_PROBE = True  # epoch 10s eeg data before probe


def feature_extract(use_ica: bool, before_probe: bool):
    """Do the feature extraction if not exist the file, else directly load the file

    Args:
        use_ica: the loaded eeg signal using ica or not
        before_probe: the epoch eeg signal is before probe (True) or before target (False)
    """
    sub_filename1 = 'ica' if use_ica else 'noica'
    sub_filename2 = 'probe' if before_probe else 'target'
    sub_filename = '{}_{}'.format(sub_filename1, sub_filename2)
    data = np.load(os.path.join(EPOCH_PATH, 'main', '{}.npz'.format(sub_filename)))

    # keys of data: eeg, rating, thought, withhold, id_list
    x = data['eeg']

    freq_domain_extract(x, sub_filename, 'PSD', fp.psd, 'on_4_bands')
    freq_domain_extract(x, sub_filename, 'SpecEnt', fp.spectral_entropy, 'on_4_bands')
    freq_domain_extract(x, sub_filename, 'STFT-SpecEnt', fp.STFT_se)

    freq_domain_extract(x, sub_filename, 'Mean', fp.mean)
    freq_domain_extract(x, sub_filename, 'MeanPower', fp.mean_power)
    freq_domain_extract(x, sub_filename, 'STD', fp.std)
    freq_domain_extract(x, sub_filename, 'FirstDiff', fp.first_diff)
    #freq_domain_extract(x, sub_filename, 'SecondDiff', fp.second_diff)
    freq_domain_extract(x, sub_filename, 'HjComp', fp.hjcomp)

    freq_domain_extract(x, sub_filename, 'WL-MeanPower', fp.WLMeanPower, 'wavelet_comp')
    freq_domain_extract(x, sub_filename, 'WL-Mean', fp.WLMean, 'wavelet_comp')
    freq_domain_extract(x, sub_filename, 'WL-STD', fp.WLstd, 'wavelet_comp')
    freq_domain_extract(x, sub_filename, 'WL-RAM', fp.WLMeanPower, 'WLRAM')
    freq_domain_extract(x, sub_filename, 'WL-Ent', fp.WLEntropy, 'wavelet_comp')
    freq_domain_extract(x, sub_filename, 'WL-SpecEnt', fp.WLSpecralEntropy, 'wavelet_comp')

    freq_domain_extract(x, sub_filename, 'MPE', fp.multiscale_permutation_entropy, 'entropy')
    freq_domain_extract(x, sub_filename, 'MSE', fp.multiscale_entropy, 'entropy')
    freq_domain_extract(x, sub_filename, 'MDE', fp.multiscale_dispersion_entropy, 'entropy')
    freq_domain_extract(x, sub_filename, 'MFDE', fp.multiscale_fluctuation_based_dispersion_entropy, 'entropy')

    freq_domain_extract(x, sub_filename, 'WL-MPE', fp.WLMPE, 'wavelet_entropy')
    freq_domain_extract(x, sub_filename, 'WL-MDE', fp.WLMDE, 'wavelet_entropy')
    freq_domain_extract(x, sub_filename, 'WL-MFDE', fp.WLMFDE, 'wavelet_entropy')


if __name__ == "__main__":
    feature_extract(use_ica=ICA, before_probe=BEFORE_PROBE)
