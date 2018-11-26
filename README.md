# OpoSum: Opinion Summarization from User Reviews

This repo will hold all code and data from the following papers:

> **Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis**,<br/>
> Stefanos Angelidis, Mirella Lapata. 2018. <br/>
> _In Transactions of the Association for Computational Linguistics (TACL)_.<br/>
> [ [pdf](http://aclweb.org/anthology/Q18-1002) ]

> **Summarizing Opinions: Aspect Extraction Meets Sentiment Prediction and They Are Both Weakly Supervised**,<br/>
> Stefanos Angelidis, Mirella Lapata. 2018. <br/>
> _In EMNLP 2018_.<br/>
> [ [pdf](http://www.aclweb.org/anthology/D18-1403) ]

*In the process of migrating data and code here.*

To train the neural models (MILNET, MATE), we first need to preprocess the raw
text files into easy to manipulate .hdf5 binaries.

First install dependencies for the preprocessing scripts:
```bash
pip install numpy --user
pip install nltk --user 
pip install h5py --user

python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

Preprocessing for MILNET:

```bash
# replace bags_and_cases with any of the domains in ./data/train
./scripts/prep_milnet.sh bags_and_cases
```
