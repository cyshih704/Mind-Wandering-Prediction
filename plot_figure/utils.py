from utils import CHANNEL_LIST, remove_people_with_same_labels
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
CHANNEL_LOCATION = np.array([[-27, 83, -3],
                             [27, 83, -3],
                             [-71, 51, -3],
                             [-48, 59, 44],
                             [0, 63, 61],
                             [48, 59, 44],
                             [71, 51, -3],
                             [-83, 27, -3],
                             [-59, 31, 56],
                             [0, 34, 81],
                             [59, 31, 56],
                             [83, 27, -3],
                             [-87, 0, -3],
                             [-63, 0, 61],
                             [0, 0, 88],
                             [63, 0, 61],
                             [87, 0, -3],
                             [-83, -27, -3],
                             [-59, -31, 56],
                             [0, -34, 81],
                             [59, -31, 56],
                             [83, -27, -3],
                             [-71, -51, -3],
                             [-48, -59, 44],
                             [0, -63, 61],
                             [48, -59, 44],
                             [71, -51, -3],
                             [-27, -83, -3],
                             [0, -87, -3],
                             [27, -83, -3]])

FONT = {'family': 'Times New Roman',
        # 'color':  'b',
        'weight': 'normal',
        'style': 'italic',
        'size': 8,
        }


def plot_rate_of_salient_features_in_each_channel(x, y):
    """plot the rate of salient features in each channel

    Args:
        :param x: # people x # trials x # channels x # features
        :param y: # people x # trials
    """

    assert len(CHANNEL_LIST) - 2 == len(CHANNEL_LOCATION)
    num_feature = x.shape[3]
    x, y = remove_people_with_same_labels(x, y, label_type='rating')

    x = x.reshape(-1, x.shape[2], x.shape[3]).transpose(1, 2, 0)  # channels x # features x (# people x # trials)
    y = y.reshape(-1)

    cm = plt.cm.get_cmap('RdYlBu')
    ch_score = []
    for ch in range(len(x)):
        num_salient_features = 0
        for f in range(len(x[ch])):
            class0_ind = y < 4
            class1_ind = y > 4
            pvalue = ttest_ind(x[ch][f][class0_ind], x[ch][f][class1_ind])[1]
            num_salient_features += int(pvalue < 0.05)
        ch_score.append(num_salient_features)
        plt.text(CHANNEL_LOCATION[ch][0]-9, CHANNEL_LOCATION[ch][1],
                 "{} ({})".format(CHANNEL_LIST[ch], num_salient_features))
        if ch == len(CHANNEL_LOCATION)-1:
            break
    sc = plt.scatter(CHANNEL_LOCATION[:, 0], CHANNEL_LOCATION[:, 1],
                     c=ch_score, cmap=cm, s=700, vmin=np.min(ch_score), vmax=np.max(ch_score))
    plt.colorbar(sc)
    plt.show()
