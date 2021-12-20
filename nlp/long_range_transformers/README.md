# Long range transformers

In recent research there has been an extensive study for improving the calculation of attention in transformer architectures. Mostly for improving their capacity to handle longer token sequences. 👊

The attention calculation is known to be quadratic in compatation time with respect to the sequence length 👎. These recent advances, however, are able to perform attention calculation in linear time with respect to the sequence length. This allows us to scale the transformer architecture such that it can handle input sequences beyond the usual token length of 512.

In this notebook, we compare traditional transformers with novel efficient transformers. We'll use roBERTa as a baseline to compare against LongFormer and BigBird.

Let's put these architectures to the test and see which one comes out on top 🏆!

We recommend to open the notebook using Colab, to check how you can finetune novel efficient transformers 👇:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/nlp/long_range_transformers/LongRangeTransformers.ipynb)
