# Mind-Wandering-Prediction

This repository aims to predict mind-wardering with EEG signals by applying entropy features on [MMSART](http://mmsart.ee.ntu.edu.tw/NTU_SART/) dataset. The experiment results show that the entropy of EEG is low when people are mind-wandering. Also, to solve the problem of high time complexity of multi-scale entropy, I find the substitution of multi-scale entropy, which is multi-scale dispersion entropy. Multi-scale dispersion entropy have similar trend as multi-scale entropy, but it has lower time complexity. The detail of this repository is [here](https://github.com/ChingYenShih/Mind-Wandering-Prediction/blob/master/presentation/mind-wandering-prediction-presentation.pdf)

## Results
Achieve **0.75** F1-score by using Fz + 2EOG channels

## Usage 

### Preprocessing
After downloading the raw data from [MMSART](http://mmsart.ee.ntu.edu.tw/NTU_SART/). Following the [preprocessing steps](https://github.com/ChingYenShih/Mind-Wandering-Prediction/tree/master/preprocessing).

### Feature Extraction
To save classification time, this step will save lots of features (power spectral density, spectral entropy, statistical features, wavelet-related features, entropy-related features). These features will be loaded when classification.
```
python3 -m feature_extraction/feature_extraction.py
```

### Classification
```
python3 classification.py
```
