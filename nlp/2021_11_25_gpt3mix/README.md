# Text Augmentation using GPT-3

The more data we have, the better performance we can achieve 🤙. However, it is sometimes difficult to annotate a large amount of training data 😞. 
Therefore, proper data augmentation is useful to boost up the model performance. Text augmentation is a very important crucial part of any NLP problem.

In this tip, we use [GPT3Mix](https://arxiv.org/pdf/2104.08826.pdf)  that leverages the GPT-3 language model to generate hyper-realistic samples from a very small dataset. GPT3Mix inputs two real samples from our dataset,embeds these samples in a carefully designed prompt and generates an augmented mixed sample influenced by the sample sentences. We use the Emotion dataset and distilled BERT pre-trained model and show that this augmentation method boosts the model performance and generates very realistic samples.

We recommend to open the notebook using Colab for an interactive explainable experience and optimal rendering of the visuals 👇:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/feature%2Fnlp_gpt3mix/nlp/2021_11_25_gpt3mix/nlp_gpt3mix.ipynb)
