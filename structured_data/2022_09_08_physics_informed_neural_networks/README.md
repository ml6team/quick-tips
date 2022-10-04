# Physics informed neural networks
Physics informed neural networks (PINNs), originating from a [paper](https://arxiv.org/pdf/1711.10561.pdf) in 2018, is a technique to solve partial differential equations through the use of machine learning. Depending on the use case, this can be significantly more efficient than use solvers such as the finite element method, as conventional solvers need to be run for each point in space and time (while a neural network is mainly compute intensive during training, but lightweight during inference).

This notebook shows how you can implement a PINN yourself in TensorFlow, and there are tools such as [DeepXDE](https://deepxde.readthedocs.io/en/latest/) that can simplify this.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/structured_data/2022_09_08_physics_informed_neural_networks/physics_informed_neural_networks.ipynb)