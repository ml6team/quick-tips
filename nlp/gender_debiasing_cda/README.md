# Gender debiasing of documents using simple CDA

## The idea

A lot of large language models are trained on webtext. However, this means that unintended biases can sneak into your model behaviour.

One such example is gender bias 👫, which can have detrimental effect on the performance of your application. Job recommendation systems for example should be as free of any gender bias as possible.

But how do we go about reducing this bias? Numerous techniques have been developed over the years, but today we will be focusing on a small and simple one: Counterfactual Data Augmentation (CDA).

Through CDA, we can swap out the gender 🔃 in a particular document, and add this debiased document as a new training example. Hopefully, this will balance out the gender-associations in the document enough to have a more suitable model 🤞!

Something like:
```
He is working as a waiter tonight, before going out to party with his sisters.
```

Should become:
```
She is working as a waitress tonight, before going out to party with her brothers.
```

## Start your engines


We recommend to open the notebook using Colab for an interactive explainable experience and optimal rendering of the visuals 👇:


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/nlp/gender_debiasing_cda/gender_debiasing_cda.ipynb)



## Results

Spoiler alert: using two measuring techniques we tried out, CDA does indeed seem to reduce the bias in the embeddings that result from a certain dataset 🥳:
![Before and after, measuring cosine similarity](images/cda_cosine_similarity.png?raw=true)
![Before and after, measuring gender vector](images/cda_gender_vector.png?raw=true)