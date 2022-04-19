# ML6 NLP Quick Tips

Current content:

-  [_Multilingual Sentence Embeddings_ (21/01/2021)](2021_01_21_multilingual_sentence_embeddings):
Gives an overview of various current multilingual sentence embedding techniques and tools, and
how they compare given various sequence lengths.

-  [_Spacy 3.0_ (01/02/2021)](2021_02_01_spacy_3_projects):
Spacy 3.0 has just been released and in this tip, we'll have a look at some of the new features.
We'll be training a German NER model and streamline the end-to-end pipeline using the brand new spaCy projects!

-  [_Compact transformers_ (26/02/2021)](2021_02_26_compact_transformers):
Bigger isn't always better. In this tip we look at some compact BERT-based models that provide a nice balance
between computational resources and model accuracy.

-  [_Keyword Extraction with pke_ (18/03/2021)](2021_03_18_pke_keyword_extraction):
The KEYNG (read *king*) is dead, long live the KEYNG!
In this tip we look at `pke`, an alternative to Gensim for keyword extraction.

-  [_Explainable transformers using SHAP_ (22/04/2021)](2021_04_22_shap_for_huggingface_transformers):
BERT, explain yourself! 📖
Up until recently language model predictions have lacked transparency. In this tip we look at `SHAP`, a way to explain your latest transformer based models.

-  [_Transformer-based Data Augmentation_ (18/06/2021)](2021_06_18_data_augmentation):
Ever struggled with having a limited non-English NLP dataset for a project? Fear not, data augmentation to the rescue ⛑️
In this week's tip, we look at backtranslation 🔀 and contextual word embedding insertions as data augmentation techniques for multilingual NLP. 

-  [_Long range transformers_ (14/07/2021)](2021_06_29_long_range_transformers):
Beyond and above the 512! 🏅 In this week's tip, we look at novel long range transformer architectures and compare them against the well-known RoBERTa model.

-  [_Neural Keyword Extraction_ (10/09/2021)](2021_09_10_neural_keyword_extraction):
Neural Keyword Extraction 🧠
In this week's tip, we look at neural keyword extraction methods and how they compare to classical methods.

-  [_HuggingFace Optimum_ (12/10/2021)](2021_10_12_huggingface_optimum):
HuggingFace Optimum Quantization ✂️
In this week's tip, we take a look at the new HuggingFace Optimum package to check out some model quantization techniques.

- [ _Text Augmentation using large-scale LMs and prompt engineering_ (25/11/2021)](2021_11_25_augmentation_lm):
Typically, the more data we have, the better performance we can achieve 🤙. However, it is sometimes difficult and/or expensive to annotate a large amount of training data 😞. In this tip, we leverage three large-scale LMs (GPT-3, GPT-J and GPT-Neo) to generate very realistic samples from a very small dataset.

- [ _Gender debaising of datasets using CDA_ (25/01/2022)](gender_debiasing_cda):
A lot of large language models are trained on webtext. However, this means that unintended biases can sneak into your model behaviour 😞. In this tip, we'll look at how to try and alleviate this bias using Counterfactual Data Augmentation ⚖️.

- [ _GPT2 Quantization using ONNXRuntime_ (19/04/2022)](gpt2_quantization):
Large language models are costly to run, in this notebook we leverage ONNXRuntime to quantize and run our Dutch GPT2 model in a more efficient way 💰.
