"""
    Epoch the EEG data 10 seconds before probe (self-rating) or target (C) appears
    Merge all subjects' data in one file
"""
import os
import sys

import numpy as np
from tqdm import tqdm

USER_NPZ_PATH = '/mnt/SART_Paper/user_preprocessed_eeg_npz'  # save processed file
EPOCH_PATH = '/mnt/SART_Paper/user_epoch_eeg_npz'  # save processed file

ICA = bool(int(sys.argv[1]))
BEFORE_PROBE = bool(int(sys.argv[1]))  # if False, epoch eeg data before C appears, used in main experiment


SAMPLE_FREQ = 1000
EPOCH_LEN = 10


def check_success_with_hold(target_idx_list, target_resp_idx_list, trial_number):
    """Return True if subject success with-hold when c appear in a trial.

    If there is a target response between 2 target -> not with-hold
    If there is no target response between 2 target -> with-hold
    """
    if trial_number != len(target_idx_list) - 1:
        pre_target_idx = target_idx_list[trial_number]
        post_target_idx = target_idx_list[trial_number + 1]

        with_hold = 1 - np.sum((target_resp_idx_list > pre_target_idx) &
                               (target_resp_idx_list < post_target_idx))
    else:  # check if the target_resp_idx appear after the last target_idx
        with_hold = 1 - np.sum(target_resp_idx_list > target_idx_list[-1])

    return with_hold


def merge_epoch_data_main():
    """Epoch eeg data and proccess label, merge data of all subjects into one file

    keys of npz file
        eeg: num of subjects x num of trials x num of channels x time
        rating: num of subjects x num of trials
        thought: num of subjects x num of trials
        withhold: num of subjects x num of trials
        id_list: num of subjects
    """
    # read preprocessed EEG data, choosing use ICA or not
    sub_dir = 'ica' if ICA else 'noica'
    file_name_list = sorted(os.listdir(os.path.join(USER_NPZ_PATH, sub_dir, 'main')))

    eeg_people, rating_people, thought_people, withhold_people = [], [], [], []
    id_people = []
    for fn in tqdm(file_name_list):
        id = fn[4:6]

        file_path = os.path.join(USER_NPZ_PATH, sub_dir, 'main', fn)
        data = np.load(file_path, allow_pickle=True)

        eeg = data['eeg']
        thought = data['thought']
        response_time = data['response_time']
        rating = data['rating']
        trigger = data['trigger']

        probe_idx_list = np.where(trigger == 'probe')[0]  # len of probe = 40
        target_idx_list = np.where(trigger == 'target')[0]  # len of target = 40, target will appear before probe
        target_resp_idx_list = np.where(trigger == 'target_response')[0]

        eeg_person, rating_perosn, thought_person, withhold_person = [], [], [], []
        for j in range(len(probe_idx_list)):  # to process data in each trial
            # with-hold or not in each trial
            with_hold = check_success_with_hold(target_idx_list, target_resp_idx_list, trial_number=j)

            if BEFORE_PROBE:  # Epoch data before self-rating
                epoch_eeg = eeg[:, probe_idx_list[j]-SAMPLE_FREQ*EPOCH_LEN: probe_idx_list[j]]
            else:  # Epoch data before C appears
                epoch_eeg = eeg[:, target_idx_list[j]-SAMPLE_FREQ*EPOCH_LEN: target_idx_list[j]]

            eeg_person.append(epoch_eeg)
            rating_perosn.append(rating[target_idx_list[j]])
            thought_person.append(thought[target_idx_list[j]])
            withhold_person.append(with_hold)

        eeg_people.append(eeg_person)
        rating_people.append(rating_perosn)
        thought_people.append(thought_person)
        withhold_people.append(withhold_person)
        id_people.append(id)

    sub_file_name1 = 'ica' if ICA else 'noica'
    sub_file_name2 = 'probe' if BEFORE_PROBE else 'target'
    saved_file_name = '{}_{}'.format(sub_file_name1, sub_file_name2)
    saved_file_path = os.path.join(EPOCH_PATH, 'main', saved_file_name)
    np.savez(saved_file_path, eeg=eeg_people, rating=rating_people,
             thought=thought_people, withhold=withhold_people, id_list=id_people)


def merge_data_rest(mode):
    sub_dir = 'ica' if ICA else 'noica'
    file_name_list = sorted(os.listdir(os.path.join(USER_NPZ_PATH, sub_dir, mode)))

    close_eeg_people, open_eeg_people, id_list = [], [], []
    for fn in tqdm(file_name_list):
        id = fn[4:6]

        file_path = os.path.join(USER_NPZ_PATH, sub_dir, mode, fn)
        data = np.load(file_path, allow_pickle=True)

        close_eeg = data['close_eeg']
        open_eeg = data['open_eeg']

        close_eeg_people.append(close_eeg[:, -3*60*SAMPLE_FREQ:])
        open_eeg_people.append(open_eeg[:, -3*60*SAMPLE_FREQ:])
        id_list.append(id)
    print(np.array(close_eeg_people).shape)
    saved_file_path = os.path.join(EPOCH_PATH, mode, sub_dir)
    np.savez(saved_file_path, close_eeg=close_eeg_people,
             open_eeg=open_eeg_people, id_list=id_list)


def main():
    merge_epoch_data_main()
    # merge_data_rest(mode='pre')
    # merge_data_rest(mode='post')


if __name__ == '__main__':
    main()
