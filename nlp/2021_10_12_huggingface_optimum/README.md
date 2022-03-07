# Huggingface Optimum

Scaling and productionizing Transformers with millions of parameters is difficult!

Addressing this, Huggingface 🤗 released a new tool called _Optimum_ (https://huggingface.co/blog/hardware-partners-program) which aims to speed up the inference time of Transformers 🏎️!

This notebook demonstrates some experiments on quantizing HF pre-trained models for sentiment analysis 🎭 and summarization 🤏.

It also compares the performance of Optimum - LPOT quantization, ONNX/ONNX Runtime quantization, and the baseline model.

It's recommended to run this notebook using Google Cloud AI Platform, using an N2-standard-4 machine. But for ease of use, you can follow this link for a Colab version 👇:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](optimum.ipynb)
