# Text Augmentation using large-scale LMs and prompt engineering

Typically, the more data we have, the better performance we can achieve 🤙. However, it is sometimes difficult and/or expensive to annotate a large amount of training data 😞. Therefore, proper data augmentation is useful to boost the model performance.

Large-scale language models (LMs) are excellent few-shot learners, allowing them to be controlled via natural text prompts. In this tip, we leverage three large-scale LMs (GPT-3, GPT-J and GPT-Neo) and prompt engineering to generate very realistic samples from a very small dataset. The model takes as input two real samples from our dataset, embeds them in a carefully designed prompt and generates an augmented mixed sample influenced by the sample sentences. We use the [Emotion](https://huggingface.co/datasets/emotion) dataset and distilled BERT pre-trained model and show that this augmentation method boosts the model performance and generates very realistic samples. For more information on text augmentation using large-scale LMs check [GPT3Mix](https://arxiv.org/pdf/2104.08826.pdf).

We recommend to open the notebook using Colab for an interactive explainable experience and optimal rendering of the visuals 👇:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/nlp/augmentation_lm/nlp_augmentation_lm.ipynb)
