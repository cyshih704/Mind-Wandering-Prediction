import numpy as np


CHANNEL_LIST = np.array(['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                         'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7',
                         'C3', 'CZ', 'C4', 'T8', 'TP7', 'CP3',
                         'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4',
                         'P8', 'O1', 'OZ', 'O2', 'HEO', 'VEO'])


def remove_people_with_same_labels(x, y, label_type: str):
    """Remove the people who annotate same labels over all trials

    Args:
        :param x: # people x # trials x # channels x # features
        :param y: # pepole x # trials
        :param label_type: {'rating', 'thought', 'withhold'}
    """
    assert label_type in {'rating', 'thought', 'withhold'}

    remove_idx = []

    y_tmp = y.copy()

    if label_type == 'rating':
        y_tmp[y_tmp < 4] = 0
        y_tmp[y_tmp > 4] = 1

        for subject in range(len(y_tmp)):
            y_sub = y_tmp[subject]
            y_sub = y_sub[y_sub != 4]

            if sum(y_sub) == 0 or sum(y_sub) == len(y_sub):
                remove_idx.append(subject)

    elif label_type == 'thought':
        raise NotImplementedError("Not implement remove people with same thought")

    else:
        for subject in range(len(y)):
            y_sub = y[subject]
            if sum(y_sub) == 0 or sum(y_sub) == len(y_sub):
                remove_idx.append(subject)

    x = np.delete(x, remove_idx, axis=0)
    y = np.delete(y, remove_idx, axis=0)

    return x, y


def specify_channels(x, log, sel_ch):
    """Specify the channel used to classification or do the further analysis

    Args:
        :param x: # people x # trials x # channels x # features
        :param log: # features
        :param sel_ch: list of channel name (str)
    Returns:
        x: # people x # trials x (# features x # specified channels)
        log: (# features x # specified channels)
    """
    # convert list of channel name (str) to list of index (int)
    sel_ch_idx = np.array([np.where(CHANNEL_LIST == ch)[0][0] for ch in sel_ch])

    x = x[:, :, sel_ch_idx, :]
    x = x.reshape(x.shape[0], x.shape[1], -1)

    # add the channel to feature log
    log_channel = []
    for i in range(len(sel_ch)):
        for j in range(len(log)):
            log_channel.append('{}_{}'.format(sel_ch[i], log[j]))

    assert x.shape[2] == len(log_channel)

    return x, np.array(log_channel)
