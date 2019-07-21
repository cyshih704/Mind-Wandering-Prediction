"""
    Save the data to npz format, and can be used for others
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, lfilter

USER_CSV_PATH = '/mnt/SART_Paper/user_eeg_csv'  # reading raw csv file
USER_NPZ_PATH = '/mnt/SART_Paper/user_preprocessed_eeg_npz'  # save processed file

PREPROCESSING = True
ICA = True

# Channel list with M2 channel
CHANNEL_LIST_WITH_M2 = np.array(['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                                 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7',
                                 'C3', 'CZ', 'C4', 'T8', 'M2', 'TP7', 'CP3',
                                 'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4',
                                 'P8', 'O1', 'OZ', 'O2', 'HEO', 'VEO'])

# Channel list without M2 channel
CHANNEL_LIST = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7',
                'C3', 'CZ', 'C4', 'T8', 'TP7', 'CP3',
                'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4',
                'P8', 'O1', 'OZ', 'O2', 'HEO', 'VEO']

# The eye component's index in each mode_subject (which is non-zero)
ICA_EXCLUDED_COMPONENTS = dict(main_26=1, main_47=1, post_03=1, post_10=1,
                               post_15=1, post_23=1, post_24=1, post_28=1,
                               post_29=3, post_30=3, post_33=8, post_43=1,
                               post_47=1, post_50=1, post_54=4, post_57=1,
                               post_65=2, post_70=2, pre_24=2, pre_33=2,
                               pre_49=1, pre_54=14, pre_65=1, pre_72=1)


def bandpass(eeg, fs=1000.0, lowcut=0.1, highcut=45.0):
    """bandpass filter from 0.1 Hz to 45 Hz

        Args:
            eeg: 1-dim list or array
            fs: sampling rate
            lowcut: lower cutoff frequency
            highcut: higher cutoff frequency

        Return:
            filtered eeg signal: 1-dim list
    """
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    return _butter_bandpass_filter(eeg, lowcut, highcut, fs, order=2)


def save_ic_component_signals(ica, raw, saved_file_path):
    """Save the projection of original EEG on each IC components

    Args:
        ica: fitted ica
        raw: raw EEG
        saved_file_path: saved file path
    """
    k = 0
    for i in range(6):
        for j in range(5):
            if i == 0 and j == 0:
                plt.figure(figsize=(19, 10))
                l = 1
            plt.subplot(6, 5, l).set_title('ICA {}'.format(k))
            plt.plot(ica.get_sources(raw).get_data()[k])
            k += 1
            l += 1
    plt.tight_layout()
    plt.savefig(saved_file_path)


def ica_remove_eye_artifact(eeg, saved_file_name):
    """Remove eye artifact of EEG using ICA

    Args:
        eeg: raw eeg (shape: (32 channels x # of time stamps))
        saved_file_name: the file name of IC components and the projection of EEG on each IC components

    Return:
        The eeg with ICA eye artifact rejection (32 channels x # of time stamps)
    """
    # NOTE: https://cbrnr.github.io/2018/01/29/removing-eog-ica/

    info = mne.create_info(CHANNEL_LIST, 1000, ch_types=["eeg"] * 30 + ["eog"] * 2)
    # sampling frequency of 1000 Hz and the channel types
    raw = mne.io.RawArray(eeg, info)

    raw.set_montage("standard_1020")
    # standard 10–20 montage, which maps the channel labels to their positions on the scalp.

    raw_tmp = raw.copy()
    raw_tmp.filter(1, None)  # apply a high-pass filter to this copy. filter < 1 Hz signals

    ica = mne.preprocessing.ICA(method="fastica", random_state=1)
    ica.fit(raw_tmp)

    save_ic_component_signals(ica, raw.copy(), os.path.join(USER_NPZ_PATH, 'ica_fig/signals/', saved_file_name))

    # plot 30 components, you have to observe where the IC component is and remove it
    ica.plot_components(inst=raw_tmp, show=False, picks=np.arange(30))
    plt.savefig(os.path.join(USER_NPZ_PATH, 'ica_fig/components/', saved_file_name))

    raw_corrected = raw.copy()

    # remove IC component, you have to check where the IC component is, it is not always at index 0
    if saved_file_name in ICA_EXCLUDED_COMPONENTS:
        print(saved_file_name, ICA_EXCLUDED_COMPONENTS[saved_file_name])
        raw_corrected = ica.apply(raw_corrected, exclude=[ICA_EXCLUDED_COMPONENTS[saved_file_name]])
    else:
        raw_corrected = ica.apply(raw_corrected, exclude=[0])
    ret = raw_corrected.get_data()

    return ret


def csv_to_npz(mode):
    """Processed raw csv file to npz format, and do the preprocessing,
    read the file from USER_CSV_PATH and save to USER_NPZ_PATH

    Preprocessing: re-reference -> bandpass filter -> ICA (optional)

    Args:
        mode: str (post, pre or main)
    """
    file_dir = os.path.join(USER_CSV_PATH, mode)
    file_names = sorted(os.listdir(file_dir))

    for fn in tqdm(file_names):
        user_id = fn[4:6]

        if int(user_id) != int(sys.argv[1]):
            continue
        saved_file_name = '{}_{}'.format(mode, user_id)
        if saved_file_name not in ICA_EXCLUDED_COMPONENTS:
            continue

        file_path = os.path.join(file_dir, fn)
        df = pd.read_csv(file_path, sep=',', low_memory=False)
        eeg = df.values  # time stamps x 38 (1 time stamp + 33 channels + 4 labels)

        if mode == 'main':
            thought, response_time, rating, trigger = eeg[:, -1], eeg[:, -2], eeg[:, -3], eeg[:, -4]
            """
                thought: normal, probe, response, target
                when thought == target, the response time, rating and trigger will have value
            """
            eeg = np.delete(eeg, [0, 34, 35, 36, 37], axis=1)  # remove time stamp and 4 labels
        else:  # resting
            trigger = eeg[: -1]
            eeg = np.delete(eeg, [0, 34], axis=1)  # remove time stamp and trigger

            close_start_idx = np.where(trigger == 'target')[0][0]  # index of close eye resting beginning
            close_end_idx = np.where(trigger == 'target_response')[0][0]  # index of close eye resting ending
            open_start_idx = np.where(trigger == 'normal')[0][0]  # index of open eye resting beginning
            open_end_idx = np.where(trigger == 'response')[0][0]  # index of open eye resting ending

        eeg = np.transpose(eeg.astype('float'))  # number of channels x time stamps

        if PREPROCESSING:

            # Re-reference
            eeg = eeg - 0.5 * eeg[np.where(CHANNEL_LIST_WITH_M2 == 'M2')[0][0]]

            # Delete referenced channel (32 x # of sample)
            eeg = np.delete(eeg, np.where(CHANNEL_LIST_WITH_M2 == 'M2')[0][0], axis=0)

            # Bandpass filter over all channels
            eeg = np.array([bandpass(ch) for ch in eeg])

        if mode != 'main':
            close_eeg = eeg[:, close_start_idx:close_end_idx]
            open_eeg = eeg[:, open_start_idx:open_end_idx]

        # Using ICA to remove eye artifact
        if PREPROCESSING and ICA:
            if mode == 'main':
                eeg = ica_remove_eye_artifact(eeg, saved_file_name='{}_{}'.format(mode, user_id))
            else:
                open_eeg = ica_remove_eye_artifact(open_eeg, saved_file_name='{}_{}'.format(mode, user_id))

        # The path of saved npz files
        sub_path = 'ica' if ICA else 'noica'
        saved_file_path = os.path.join(USER_NPZ_PATH, sub_path, mode, 'user'+user_id)

        # Save the data to npz files
        if mode == 'main':
            np.savez(saved_file_path, eeg=eeg, thought=thought,
                     response_time=response_time, rating=rating, trigger=trigger)
        else:
            np.savez(saved_file_path, open_eeg=open_eeg, close_eeg=close_eeg)


def main():
    """
        main function
    """
    assert PREPROCESSING == True

    csv_to_npz('main')
    csv_to_npz('post')
    csv_to_npz('pre')


if __name__ == '__main__':
    main()
