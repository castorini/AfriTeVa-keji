# AfriTeVa-Keji

This repository contains code to reproduce [Better Quality Pre-training Data and T5 Models for African Languages]() which appeats in the 2023 conference on [Empirical Methods in Natural Language Processing (EMNLP)](https://2023.emnlp.org/).

AfriTeVa-Keji was trained on 20 languages (16 African Langauges + Arabic, English, French, Portuguese) as the successor to [AfriTeVa](https://github.com/castorini/afriteva) and evaluated on text classification, summarisation, reading comprehension and machine translation.

## Setup

* Create a conda environment. Note that this repo has only been tested with Python=3.9. For 

```bash
conda create -n teva python=3.9 -y
```

* Install JAX and t5x for your device.

```bash
# For TPU
pip install -r requirements/requirements-tpu.txt
```

```bash
# For GPU. Note that this installs jax for CUDA 12
# For other CUDA versions, you may need to edit the requirements/requirements-gpu.txt
pip install -r requirements/requirements-gpu.txt
```

* Install `teva`. 

```bash
# For normal installation
pip install .
```

```bash
# For development installation
pip install -e .
```

## Experiments

### Datasets

AfriTeVa-Keji was pretrained on the Wúrà dataset which is available through Huggingface Hub [here]().

### Language Modelling

```bash
bash scripts/teva_utils.sh --pretrain --size base
```

### Text Classification

AfriTeVa-Keji was evaluated on [MasakhaNEWS 2.0](https://github.com/masakhane-io/masakhane-news) which covers 16 langauges widely spoken in Africa.

```bash
# This will train a single classifier for the specified language
bash scripts/teva_utils --train-classifier --mode monolingual --language amh
```

```bash
# This will train a single classifier for all languages
bash scripts/teva_utils.sh --train-classifier --mode multilingual
```

### Summarisation

AfriTeVa-Keji was evaluated on 15 of the languages in [XL-SUM](https://github.com/csebuetnlp/xl-sum) 

### Machine Translation

TODO.

### Reading Comprehension

TODO.

## Citation

TODO.
