# AfriTeVa-Keji

This repository contains code to reproduce [Better Quality Pre-training Data and T5 Models for African Languages](https://openreview.net/forum?id=ybc9V6Cbq2) which appeats in the 2023 conference on [Empirical Methods in Natural Language Processing (EMNLP)](https://2023.emnlp.org/).

AfriTeVa V2 was trained on 20 languages (16 African Langauges + Arabic, English, French, Portuguese) as the successor to [AfriTeVa](https://github.com/castorini/afriteva) and evaluated on text classification, summarisation, reading comprehension and machine translation.

We release the following models:

* [AfriTeVa V2 Base (428M)](https://huggingface.co/castorini/afriteva_v2_base)
* [AfriTeVa V2 Large (1B)](https://huggingface.co/castorini/afriteva_v2_large)
## Setup

* Create a conda environment. Note that this repo has only been tested with Python 3.9.

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

There are a few environment variables you may need to set. See [.example.env](.example.env).

## Experiments

### Datasets

AfriTeVa-Keji was pretrained on the Wúrà dataset which is available through Huggingface Hub [here](https://huggingface.co/datasets/castorini/wura).


### Language Modelling

To pretrain AfriTeVa V2, simply follow the setup instructions

```bash
bash scripts/pretrain.sh
```

If you need to convert the t5x checkpoint to flax, run the following command

```bash
python -m transformers.models.t5x.convert_t5_checkpoint_to_flax \
--t5x_checkpoint_path /path/to/your/trained/base/model \
--config_name config/models/t5_1_1/base.json \
--flax_dump_folder_path /path/to/your/converted/model
```

### Text Classification

AfriTeVa-Keji was evaluated on [MasakhaNEWS 2.0](https://github.com/masakhane-io/masakhane-news) which covers 16 languages widely spoken in Africa.

```bash
# This will train and evaluate a classifier for each language over three seeds.
bash scripts/tasks/masahanews_ft.sh
```

### Summarisation

AfriTeVa-Keji was evaluated on 15 of the languages in [XL-SUM](https://github.com/csebuetnlp/xl-sum) 

```bash
# This will perform multilingual finetuning over 50,000 steps.
bash scripts/tasks/xlsum_xlingual.sh
```

### Machine Translation

AfriTeVa V2 was evaluated on [MAFAND-MT](https://huggingface.co/datasets/masakhane/mafand). 

```bash
bash scripts/tasks/lafand_mt.sh
```

## Citation

```bibtex
@article{OladipoBQPD2023EMNLP,
  title = "Better Quality Pre-training Data and T5 Models for African Languages",
  author = "Oladipo, Akintunde and
    Adeyemi, Mofetoluwa and
    Ahia, Orevaoghene and
    Owodunni, Abraham and
    Ogundepo, Odunayo and
    Adelani, David and
    Lin, Jimmy
  ",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  publisher = "Association for Computational Linguistics",
  year = "2023",
}
```
