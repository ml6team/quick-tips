# Time series: classification with MINIROCKET
For time series classification, we usually rely on some sort of feature extraction combined with a traditional classifier. While the _tsfresh_ package is often preferred for the former task, it becomes exponentially slow for growing dataset sizes thus making it cumbersome or even impossible to use in real-life use cases. 

In a publication preprint by _A. Dempster et al._, [_MINIROCKET: A Very Fast (Almost) Deterministic Transform
for Time Series Classification_](https://arxiv.org/pdf/2012.08791.pdf), an alternative kernel-based feature transformation method is proposed. It can extract features in both CPU and GPU environments at a fraction of _tsfresh_'s efficient or comprehensive feature calculator settings execution time (especially for large datasets), achieving a similar accuracy  but at the cost of features that are not at all interpretable. 
Additionally, while it is not as fast as the almost instantaneous Fourier transformed (_fft_/_stft_) features, _MINIROCKET_  generally offers better performance.

Open the notebook in Colab to learn how to use this method in CPU and GPU environment and to witness how it compares to our current way of working:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/structured_data/2021_06_04_minirocket_timeseries_classification/minirocket_timeseries_classification.ipynb)
