# TensorFlow Data Validation
TensorFlow Data Validation (TFDV) is a tool to allow that allows you to
detect discrepancies between datasets. Its functionality includes:
* Visualizing your data.
* Automatically inferring a schema for your data.
* Detecting when a dataset doesn't match the schema.
* Detecting when different datasets diverge too much.

This chapter tip goes into the standalone version of TFDV.

TFDV can also be used as part of TFX (more info here: https://www.tensorflow.org/tfx/guide#data_exploration_visualization_and_cleaning).

Vertex AI has a managed version of TFDV (more info here: https://cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring).

The notebook found in this folder goes over the different functionalities and how to use it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/structured_data/2021_06_30_tensorflow_data_validation/tfdv_functionality_showcase.ipynb)

## Measuring distribution divergence
The notebook goes over the skew and drift comparators using Jenssen Shannon Divergence (for numerical features) 
and the L_\infty norm (for categorical features) as divergence metrics.

To learn more about these metrics, we refer to the following sources:
* [Jensen-Shannon diverce](https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15)
* [L<sub>∞</sub> - norm](https://en.wikipedia.org/wiki/Chebyshev_distance)