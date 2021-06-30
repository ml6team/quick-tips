# TensorFlow Data Validation
TensorFlow Data Validation (TFDV) is a tool to allow that allows you to
detect discrepancies between datasets. Its functionality includes:
* Visualizing your data
* Automatically inferring a schema for your data
* Detecting when a dataset doesn't match the schema
* Detecting when different datasets diverge too much

This chapter tip goes into the standalone version of TFDV.
In the accompanying chapter talk (link to video), we go deeper into
data drift & skew, and how the different divergence metrics are defined.

TFDV can also be used as part of TFX (more info here: https://www.tensorflow.org/tfx/guide#data_exploration_visualization_and_cleaning).

Vertex AI has a managed version of TFDV (more info here: https://cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring).

The notebook found in this folder goes over the different functionalities and how to use it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/structured_data/2021_06_30_tensorflow_data_validation/tfdv_functionality_showcase.ipynb)