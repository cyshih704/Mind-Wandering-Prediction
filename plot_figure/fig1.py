from feature_extraction.feature_extraction import feature_extract
from plot_figure.utils import plot_rate_of_salient_features_in_each_channel
from utils import normalize
ICA = False  # Use ICA or not
BEFORE_PROBE = True  # epoch 10s eeg data before probe
LABEL_TYPE = 'rating'

if __name__ == "__main__":
    x, y, log = feature_extract(use_ica=ICA, before_probe=BEFORE_PROBE, label_type=LABEL_TYPE)
    #x = normalize(x)
    plot_rate_of_salient_features_in_each_channel(x, y)
