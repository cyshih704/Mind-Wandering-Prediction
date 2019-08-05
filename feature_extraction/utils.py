import os
from tqdm import tqdm
import numpy as np

FEATURE_PATH = '/home/access/cys/SART_Paper/feature_npz'  # saved feature path


def freq_domain_extract(x, sub_dir: str, feature_name: str, function, extract_type=None):
    """Do the frequency domain feature extraction

    Args:
        x: eeg (shape: # people x # trials x # channels x # samples)
    """

    saved_dir = os.path.join(FEATURE_PATH, sub_dir)
    if not os.path.exists(os.path.join(saved_dir)):
        os.makedirs(os.path.join(saved_dir))

    saved_file_path = os.path.join(saved_dir, feature_name)

    if os.path.isfile(saved_file_path+'.npz'):  # if already extract the feature, load directly
        print('    Loading [{}] from [{}]'.format(feature_name, sub_dir))

        data = np.load(saved_file_path+'.npz')
        x = data['x']
        log = data['log']
    else:
        print('    Extracting [{}] on [{}]'.format(feature_name, sub_dir))

        pbar = tqdm(range(len(x)))
        x_subject = []
        for subject in pbar:
            x_trial = []
            for trial in range(len(x[subject])):
                x_channel = []
                for channel in range(len(x[subject][trial])):
                    x_in = x[subject][trial][channel]
                    if extract_type == 'on_4_bands':
                        feature, log = freq_domain(x_in, function, feature_name)
                    elif extract_type == 'wavelet_comp':
                        feature, log = wavelet_domain(x_in, function, feature_name)
                    elif extract_type == 'entropy':
                        feature, log = entropy_domain(x_in, function, feature_name)
                    elif extract_type == 'wavelet_entropy':
                        feature, log = wavelet_entropy_domain(x_in, function, feature_name)
                    else:
                        feature, log = time_domain(x_in, function, feature_name)

                    x_channel.append(feature)

                x_trial.append(x_channel)

            x_subject.append(x_trial)
            pbar.set_description("Extract {} from subject {}".format(feature_name, subject+1))
            # if subject == 0:
            #    break

        x_subject = np.array(x_subject)
        log = np.array(log)
        assert x_subject.shape[3] == len(log)

        np.savez(saved_file_path, x=x_subject, log=np.array(log))
        print('         Save to {}'.format(saved_file_path))
        print('         feature_dim: {}'.format(x_subject.shape))
        print('         feature_log: {}'.format(log))


def freq_domain(x, function, feature_name):
    theta = [4, 7]
    alpha = [8, 13]
    beta = [14, 29]
    gamma = [30, 45]
    bands = [theta, alpha, beta, gamma]
    bands_log = ['theta', 'alpha', 'beta', 'gamma']

    feature = []
    log = []
    for i, band in enumerate(bands):
        feature.append(function(x=x,
                                start_freq=band[0],
                                end_freq=band[1],
                                fs=1000.0))
        log.append('{}_{}'.format(feature_name, bands_log[i]))
    return feature, log


def time_domain(x, function, feature_name):
    feature = [function(x)]
    log = [feature_name]
    return feature, log


def wavelet_domain(x, function, feature_name):
    if feature_name == 'WLRAM':
        components = ['cA7cD7', 'cD7cD6', 'cD6cD6', 'cD5cD4']
    else:
        components = ['cA7', 'cD7', 'cD6', 'cD5', 'cD4']

    feature = function(x)
    log = ['{}-{}'.format(comp, feature_name) for comp in components]
    return feature, log


def wavelet_entropy_domain(x, function, feature_name):
    feature = function(x)
    log = ['{}-{}'.format(feature_name, i+1) for i in range(len(feature))]
    return feature, log


def wavelet_entropy_domain(x, function, feature_name):
    components = ['cA7', 'cD7', 'cD6', 'cD5', 'cD4']

    feature = function(x)
    log = []
    for comp in components:
        for i in range(20):
            log.append('{}-{}-{}'.format(feature_name, comp, i+1))

    return feature, log
