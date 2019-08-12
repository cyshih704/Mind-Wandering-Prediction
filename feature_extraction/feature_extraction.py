import os

import numpy as np
from tqdm import tqdm

import feature_extraction.feature_pool as fp
from feature_extraction.utils import extract_one_feature_and_save

EPOCH_PATH = '/home/access/cys/SART_Paper/user_epoch_eeg_npz'  # Load the epoch data
FEATURE_PATH = '/home/access/cys/SART_Paper/feature_npz'  # saved feature path

ICA = False  # Use ICA or not
BEFORE_PROBE = True  # epoch 10s eeg data before probe


def feature_extract(use_ica: bool, before_probe: bool, label_type: str):
    """Do the feature extraction if not exist before, or directly load the file
       Convert the label to binary

    Args:
        use_ica: the loaded eeg signal using ica (True) or not (False)
        before_probe: the epoch eeg signal is before probe (True) or before target (False)
        label_type: {'rating', 'thought', 'withhold'}
    Returns:
        feature: # people x # trials x # channels x # features
        label: # pepole x # trials
        feature log: # features (the name of feature)
    """
    sub_filename1 = 'ica' if use_ica else 'noica'
    sub_filename2 = 'probe' if before_probe else 'target'
    sub_filename = '{}_{}'.format(sub_filename1, sub_filename2)
    data = np.load(os.path.join(EPOCH_PATH, 'main', '{}.npz'.format(sub_filename)))

    # keys of data: eeg, rating, thought, withhold, id_list
    x = data['eeg']
    # x = None
    y = data[label_type]

    psd, psd_log = extract_one_feature_and_save(x, sub_filename, 'PSD', fp.psd, 'freq_band')
    se, se_log = extract_one_feature_and_save(x, sub_filename, 'SpecEnt', fp.spectral_entropy, 'freq_band')
    stft_se, stft_se_log = extract_one_feature_and_save(x, sub_filename, 'STFT-SpecEnt', fp.STFT_se)

    mean, mean_log = extract_one_feature_and_save(x, sub_filename, 'Mean', fp.mean)
    mean_power, mean_power_log = extract_one_feature_and_save(x, sub_filename, 'MeanPower', fp.mean_power)
    std, std_log = extract_one_feature_and_save(x, sub_filename, 'STD', fp.std)
    first_diff, first_diff_log = extract_one_feature_and_save(x, sub_filename, 'FirstDiff', fp.first_diff)
    # extract_one_feature_and_save(x, sub_filename, 'SecondDiff', fp.second_diff)
    hjcomp, hjcomp_log = extract_one_feature_and_save(x, sub_filename, 'HjComp', fp.hjcomp)

    wl_mean_power, wl_mean_power_log = extract_one_feature_and_save(
        x, sub_filename, 'WL-MeanPower', fp.wl_mean_power, 'wavelet_comp')
    wl_mean, wl_mean_log = extract_one_feature_and_save(x, sub_filename, 'WL-Mean', fp.wl_mean, 'wavelet_comp')
    wl_std, wl_std_log = extract_one_feature_and_save(x, sub_filename, 'WL-STD', fp.wl_std, 'wavelet_comp')
    wl_ratio_of_subbands, wl_ratio_of_subbands_log = extract_one_feature_and_save(
        x, sub_filename, 'WL-RAM', fp.wl_ratio_of_subbands, 'wavelet_comp')
    wl_entropy, wl_entropy_log = extract_one_feature_and_save(x, sub_filename, 'WL-Ent', fp.wl_entropy, 'wavelet_comp')
    wl_spec_entropy, wl_spec_entropy_log = extract_one_feature_and_save(
        x, sub_filename, 'WL-SpecEnt', fp.wl_spec_entropy, 'wavelet_comp')

    mse, mse_log = extract_one_feature_and_save(x, sub_filename, 'MSE', fp.multiscale_entropy, 'entropy')
    mpe, mpe_log = extract_one_feature_and_save(x, sub_filename, 'MPE', fp.multiscale_permutation_entropy, 'entropy')
    mde, mde_log = extract_one_feature_and_save(x, sub_filename, 'MDE', fp.multiscale_dispersion_entropy, 'entropy')
    mfde, mfde_log = extract_one_feature_and_save(
        x, sub_filename, 'MFDE', fp.multiscale_fluctuation_based_dispersion_entropy, 'entropy')

    wl_mpe, wl_mpe_log = extract_one_feature_and_save(x, sub_filename, 'WL-MPE', fp.wl_mpe, 'wavelet_entropy')
    wl_mde, wl_mde_log = extract_one_feature_and_save(x, sub_filename, 'WL-MDE', fp.wl_mde, 'wavelet_entropy')
    wl_mfde, wl_mfde_log = extract_one_feature_and_save(x, sub_filename, 'WL-MFDE', fp.wl_mfde, 'wavelet_entropy')

    time_feature = np.concatenate((mean, mean_power, std, first_diff, hjcomp), axis=3)
    time_feature_log = np.concatenate((mean_log, mean_power_log, std_log, first_diff_log, hjcomp_log), axis=0)
    freq_feature, freq_feature_log = psd, psd_log
    wavelet_feature = np.concatenate((wl_mean_power, wl_mean, wl_std, wl_ratio_of_subbands), axis=3)
    wavelet_feature_log = np.concatenate((wl_mean_power_log, wl_mean_log, wl_std_log, wl_ratio_of_subbands_log), axis=0)

    ent_feature_nomse = np.concatenate((mpe, mfde, mde), axis=3)
    ent_feature_nomse_log = np.concatenate((mpe_log, mfde_log, mde_log), axis=0)
    freq_ent_feature = np.concatenate((se, stft_se), axis=3)
    freq_ent_feature_log = np.concatenate((se_log, stft_se_log), axis=0)
    wavelet_ent_feature = np.concatenate((wl_mpe, wl_mde, wl_mfde, wl_entropy, wl_spec_entropy), axis=3)
    wavelet_ent_feature_log = np.concatenate(
        (wl_mpe_log, wl_mde_log, wl_mfde_log, wl_entropy_log, wl_spec_entropy_log), axis=0)

    feature = np.concatenate((time_feature, freq_feature, wavelet_feature,
                              ent_feature_nomse, freq_ent_feature, wavelet_ent_feature), axis=3)
    feature_log = np.concatenate((time_feature_log, freq_feature_log,
                                  wavelet_feature_log, ent_feature_nomse_log, freq_ent_feature_log, wavelet_ent_feature_log), axis=0)

    #feature = psd
    #feature_log = psd_log

    return feature, y, feature_log


if __name__ == "__main__":
    feature_extract(use_ica=ICA, before_probe=BEFORE_PROBE, label_type='rating')
