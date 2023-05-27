# AfriTeVa-keji


## Setup

### t5x Installation

* Create a conda environment. This repo has only been tested with Python=3.8

```bash
conda create -n teva python=3.8
```

* Basic development installation command

```bash
pip install -e ".[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"    # For TPU
pip install -e ".[gpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html      # For GPU
```

* For GPU, you may additionally need to install the JAX version compatible with your CUDA. For CUDA 12, this command will be:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### t5-experiments Installation

TODO