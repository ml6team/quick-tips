# The KEYNG (read *king*) is dead, long live the KEYNG!

For a long time the go-to library for graph-based keyword extraction has been Gensim. However, the next major release—version 4.0 which is currently in beta—will [remove the entire summarisation module](https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4#12-removed-gensimsummarization), which includes the keyword extraction functionality.

This motivates an update on **keyword extraction** from a previous Tip of the week on extractive summarization which included a section about keyword extraction with Gensim.

This notebook gives a brief overview of **pke** an **open-source keyphrase extraction toolkit**, that is easy to use, provides a wide range of keyword extraction methods which makes it easy to benchmark different approaches in order to choose the right algorithm for the problem at hand.


We recommend to open the notebook using Colab to extract keywords using `pke` yourself 👇:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml6team/quick-tips/blob/main/nlp/pke_keyword_extraction/pke_keyword_extraction.ipynb)
