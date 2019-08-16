from feature_extraction.feature_extraction import feature_extract
from plot_figure.utils import plot_rate_of_salient_features_in_each_channel, plot_entropy_scale_value_on_diff_class
from utils import normalize

ICA = False  # Use ICA or not
BEFORE_PROBE = True  # epoch 10s eeg data before probe
LABEL_TYPE = 'rating'

if __name__ == "__main__":
    x, y, log = feature_extract(use_ica=ICA, before_probe=BEFORE_PROBE, label_type=LABEL_TYPE)
    #x = normalize(x)

    # fig1
    #plot_rate_of_salient_features_in_each_channel(x, y)

    # fig2
    #plot_entropy_scale_value_on_diff_class(x, y, log, entropy_log='MSE', channel='FZ')
    #plot_entropy_scale_value_on_diff_class(x, y, log, entropy_log='MPE', channel='FZ')
    #plot_entropy_scale_value_on_diff_class(x, y, log, entropy_log='MDE', channel='FZ')
    #plot_entropy_scale_value_on_diff_class(x, y, log, entropy_log='MFDE', channel='FZ')

    # fig3
    print(log)
