from feature_extraction.feature_extraction import feature_extract
from utils import remove_people_with_same_labels, specify_channels, leave_one_subject_out

ICA = False  # Use ICA or not
BEFORE_PROBE = True  # epoch 10s eeg data before probe
LABEL_TYPE = 'rating'
SEL_CH = ['FZ', 'FP1', 'FP2']
#SEL_CH = ['FZ']

if __name__ == '__main__':
    x, y, log = feature_extract(use_ica=ICA, before_probe=BEFORE_PROBE, label_type=LABEL_TYPE)
    x, y = remove_people_with_same_labels(x, y, label_type=LABEL_TYPE)
    x, log = specify_channels(x, log, SEL_CH)

    leave_one_subject_out(x=x, y=y, log=log, label_type=LABEL_TYPE)
