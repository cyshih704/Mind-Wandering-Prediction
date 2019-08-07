import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp

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


def normalize(x):
    """Normalize the features of each subject

    Args:
        :param x: # people x # trials x (# channels x # features)
    Return:
        normalized features
    """
    x_norm = []
    for subject in range(len(x)):
        x_subject_t = np.transpose(x[subject])  # num_feature x trial
        tmp = [(feature-np.mean(feature))/np.std(feature) for feature in x_subject_t]
        x_norm.append(np.transpose(np.array(tmp)))
    return np.array(x_norm)


def leave_one_subject_out(x, y, log, label_type):
    """Normalization and Leave one subject out cross validation

    Args:
        :param x: # people x # trials x (# channels x # features)
        :param y: # people x # trials
        :param log: (# channels x # features)
    """
    x = normalize(x)

    clf = XGBClassifier(n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        min_child_weight=2,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=1,
                        reg_alpha=1,
                        )
    precision_list, recall_list, f1_list = list(), list(), list()
    tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
    for subject in range(len(x)):
        x_train, y_train = np.delete(x, subject, axis=0), np.delete(y, subject, axis=0)
        x_test, y_test = x[subject], y[subject]

        # reshape x and y, and convert label to binary and remove threshold
        x_train, y_train = convert_to_binary_label_and_remove_threshold(
            x_train.reshape(-1, x.shape[2]), y_train.reshape(-1), label_type)
        x_test, y_test = convert_to_binary_label_and_remove_threshold(
            x_test.reshape(-1, x.shape[2]), y_test.reshape(-1), label_type)

        # train and predict
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # plot roc curve of each fold (subject)
        probas_ = clf.predict_proba(x_test)  # shape: len x 2 (prob of neg, prob of pos)
        fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold {} (AUC={:.2f})'.format(subject, roc_auc))

        # be used to plot mean roc
        tprs.append(interp(mean_fpr, fpr, tpr))  # append mean tpr (interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # mean_tpr[0] = 0
        aucs.append(roc_auc)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        print('Test on subject {}: Precision->{:.2f}, Recall->{:.2f}, F1->{:.2f}'.format(subject+1, precision, recall, f1))

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('---------------------------')
    print('Average: Precision->{:.2f}, Recall->{:.2f}, F1->{:.2f}'.format(np.mean(precision_list),
                                                                          np.mean(recall_list), np.mean(f1_list)))

    # plot mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' %
             (mean_auc, std_auc), lw=2, alpha=.8)

    # plot chance level roc
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)  # plot chance level ROC

    plt.show()


def leave_one_trial_out(x, y, log, label_type):
    """Normalization and Leave one subject out cross validation

    Args:
        :param x: # people x # trials x (# channels x # features)
        :param y: # people x # trials
        :param log: (# channels x # features)
    """


def convert_to_binary_label_and_remove_threshold(x, y, label_type):
    """convert label to binary, and remove threshold (4 in rating)

    Args:
        x: trials x features
        y: trials
    Returns:
        x: trials x features (remove threshold)
        y: trials (binary, and remove threshold)
    """
    assert label_type in {'rating', 'thought', 'withhold'}

    if label_type == 'rating':
        idx_4 = np.where(y == 4)[0]
        y[y < 4] = 0
        y[y > 4] = 1
        x = np.delete(x, idx_4, axis=0)
        y = np.delete(y, idx_4, axis=0)
    elif label_type == 'thought':
        raise NotImplementedError('Not implement convert_to_binary_label_and_remove_threshold of thought')
    else:
        pass
    return x, y
