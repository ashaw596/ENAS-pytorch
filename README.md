# Efficient Neural Architecture Search (ENAS) in PyTorch

PyTorch implementation of [Efficient Neural Architecture Search via Parameters Sharing](https://arxiv.org/abs/1802.03268).

<p align="center"><img src="assets/ENAS_rnn.png" alt="ENAS_rnn" width="60%"></p>

**ENAS** reduce the computational requirement (GPU-hours) of [Neural Architecture Search](https://arxiv.org/abs/1611.01578) (**NAS**) by 1000x via parameter sharing between models that are subgraphs within a large computational graph. SOTA on `Penn Treebank` language modeling.

**\*\*[Caveat] Use official code from the authors: [link](https://github.com/melodyguan/enas)\*\***


## Prerequisites

- Python 3.6+
- [PyTorch](http://pytorch.org/)
- tqdm, scipy, imageio

## Usage

Install prerequisites with:

    conda install graphviz
    pip install -r requirements.txt

## Results

Efficient Neural Architecture Search (**ENAS**) is composed of two sets of learnable parameters, controller LSTM *θ* and the shared parameters *ω*. These two parameters are alternatively trained and only trained controller is used to derive novel architectures.

### 1. Discovering Convolutional Neural Networks

(in progress)


### 2. Designing Convolutional Cells

(in progress)


## Reference

- [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
- [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
