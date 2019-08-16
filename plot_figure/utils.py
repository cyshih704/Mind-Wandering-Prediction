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


def plot_entropy_scale_value_on_diff_class(x, y, log, entropy_log, channel):
    """plot scale v.s. value diagram on diff class (for entropy feature)

    Args:
        :param x: # people x # trials x # channels x # features
        :param y: # people x # trials
        :param log: # features
        :param entropy_log: type of entropy_log = {'MSE', 'MPE', 'MDE', 'MFDE'}
        :channel: select the channel which used to plot the diagram (channel is in the CHANNEL_LIST)
    """

    assert entropy_log in {'MSE', 'MPE', 'MDE', 'MFDE'}
    assert channel in CHANNEL_LIST

    title_dic = dict(MSE='Multisclae Entropy', MPE='Multisclae Permutation Entropy',
                     MDE='Multisclae Dispersion Entropy', MFDE='Fluctuation-based Multisclae Dispersion Entropy')

    x, y = remove_people_with_same_labels(x, y, label_type='rating')

    index_of_entropy_feature = [i for i in range(len(log)) if log[i][:len(entropy_log)] == entropy_log]
    assert len(index_of_entropy_feature) == 20

    index_of_channel = np.where(CHANNEL_LIST == channel)[0][0]

    x = x[:, :, index_of_channel, index_of_entropy_feature]  # people x # trials x # scales
    x = x.reshape(-1, x.shape[2])  # (# people x # trials) x # scales
    y = y.reshape(-1)  # (# people x # trials)
    num_scale = x.shape[1]

    pos_class_index = np.where(y > 4)[0]
    neg_class_index = np.where(y < 4)[0]

    x_pos = x[pos_class_index]
    x_neg = x[neg_class_index]

    scale_with_salient_p_value = [scale+1 for scale in range(
        num_scale) if ttest_ind(x_neg[:, scale], x_pos[:, scale])[1] < 0.05]

    pos_mean_value_list = np.array([np.mean(x_pos[:, scale]) for scale in range(num_scale)])
    pos_std_value_list = np.array([np.std(x_pos[:, scale]) for scale in range(num_scale)])

    neg_mean_value_list = np.array([np.mean(x_neg[:, scale]) for scale in range(num_scale)])
    neg_std_value_list = np.array([np.std(x_neg[:, scale]) for scale in range(num_scale)])

    plt.errorbar(np.arange(1, 1+num_scale), pos_mean_value_list, yerr=pos_std_value_list, ecolor='b',
                 xuplims=True, xlolims=True, fmt='bo-', elinewidth=2, capsize=4, label='not MW')
    plt.errorbar(np.arange(1, 1+num_scale), neg_mean_value_list, yerr=neg_std_value_list, ecolor='r',
                 xuplims=True, xlolims=True, fmt='ro-', elinewidth=2, capsize=4, label='MW')

    # the y axis of star
    star_y_axis = max(np.max(pos_mean_value_list+pos_std_value_list),
                      np.max(neg_mean_value_list+neg_std_value_list))+0.2
    #star_y_axis = 4.7
    # plot the star on scale with p value < 0.05
    plt.scatter(scale_with_salient_p_value, [star_y_axis]*len(scale_with_salient_p_value), marker="*")

    plt.xticks(np.linspace(1, num_scale, num_scale))
    plt.xlabel('Scale')
    plt.ylabel('{} value'.format(entropy_log))
    plt.title(title_dic[entropy_log])
    #plt.ylim(0.3, 4.8)
    plt.legend(loc='lower right')
    plt.show()
