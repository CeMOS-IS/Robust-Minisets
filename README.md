# Robust-Minisets
[![arxiv.org](https://img.shields.io/badge/cs.CV-arXiv%3A0000.0000-B31B1B.svg)](https://arxiv.org/)
[![cite-bibtex](https://img.shields.io/badge/Cite-BibTeX-1f425f.svg)](#citing)
[![data](https://img.shields.io/badge/Download-Data-green)](https://doi.org/10.5072/zenodo.91410)
[![](https://colab.research.google.com/assets/colab-badge.svg)]()

We introduce `Robust-Minisets`, a collection of robust benchmark classification datasets in the low resolution realm based on well-established image classification benchmarks, such as CIFAR, Tiny ImageNet, EuroSAT and the MedMNIST collection. We port existing robustness and generalization benchmarks (ImageNet-C, -R, -A and v2) to the small dataset domain introducing novel benchmarks to comprehensively evaluate the robustness and generalization capabilities of image classification models on low resolution datsets. This results in an extensive collection consisting of already existing test sets (e.g. CIFAR-10.1 and Tiny ImageNet-C) as well as the novel benchmarks EuroSAT-C, MedMNIST-C, and Tiny ImageNet-A, -R and -v2 introduced in our ICPR2024 paper ["GenFormer - Generated Images are All You Need to Improve Robustness of Transformers on Small Datasets"](https://github.com/CeMOS-IS/GenFormer).

[Nikolas Ebert](https://scholar.google.de/citations?user=CfFwm1sAAAAJ&hl=de), Sven Oehri, Ahmed Abdullah, [Didier Stricker](https://scholar.google.de/citations?user=ImhXfxgAAAAJ&hl=de) & [Oliver Wasenmüller](https://scholar.google.de/citations?user=GkHxKY8AAAAJ&hl=de) \
**[CeMOS - Research and Transfer Center](https://www.cemos.hs-mannheim.de/ "CeMOS - Research and Transfer Center"), [University of Applied Sciences Mannheim](https://www.english.hs-mannheim.de/the-university.html "University of Applied Sciences Mannheim")**

<div align="left">

  <div align="center">
    <img  src="assets/montage.png" width="1000"/>
    <div>&nbsp;</div>
    
  </div>
  <div>&nbsp;</div>
  
# Code Structure
* [`robust_minisets/`](robust_minisets/):
    * [`dataset.py`](robust_minisets/dataset.py): PyTorch datasets and dataloaders of Robust-Minisets.
    * [`info.py`](robust_minisets/info.py): Dataset information `dict` for each subset of Robust-Minisets.
* [`examples/`](examples/):
    * [`getting_started.ipynb`](examples/getting_started.ipynb): To explore the Robust-Minisets dataset collection with jupyter notebook. It is ONLY intended for a quick exploration, i.e., it does not provide full training and evaluation functionalities. 
    * [`getting_started_without_PyTorch.ipynb`](examples/getting_started_without_PyTorch.ipynb): This notebook provides snippets about how to use Robust-Minisets data (the `.npz` files) without PyTorch.
* [`setup.py`](setup.py): To install `robust_minisets` as a module

# Installation and Requirements
Setup the required environments and install `robust-minisets` as a standard Python package from [PyPI](https://pypi.org/project/robust-minisets/):

    pip install -r requirements.txt
    pip install robust-minisets

Or install from source:

    pip install -r requirements.txt
    pip install --upgrade git+https://github.com/CeMOS-IS/Robust-Minisets.git

Check whether you have installed the latest code [version](robust_minisets/info.py#L1):

    >>> import robust_minisets
    >>> print(robust_minisets.__version__)

The code requires only common Python environments for machine learning. Basically, it was tested with
* Python 3 (>=3.8)
* torch, torchvision, numpy, Pillow, scikit-learn, scikit-image, tqdm, fire

Higher (or lower) versions should also work (perhaps with minor modifications). 

# Quick Start

To use a standard test set utilizing the downloaded files:

    >>> from robust_minisets import TinyImageNetR
    >>> test_dataset = TinyImageNetR(split="test")

To enable automatic downloading by setting `download=True`:

    >>> from robust_minisets import BreastMNISTC
    >>> val_dataset = BreastMNISTC(split="val", download=True)

Certain datasets (Tiny ImageNet, EuroSAT) are implemented as training datasets as well:

    >>> from robust_minisets import EuroSAT
    >>> train_dataset = EuroSAT(split="train", download=True)

## If you use PyTorch...

* Great! Our code is designed to work with PyTorch.

* Explore the Robust-Minisets dataset with jupyter notebook ([`getting_started.ipynb`](examples/getting_started.ipynb)), and train basic neural networks in PyTorch.


## If you do not use PyTorch...

* Although our code is tested with PyTorch, you are free to parse them with your own code (without PyTorch or even without Python!), as they are only standard NumPy serialization files. It is simple to create a dataset without PyTorch.
* Go to [`getting_started_without_PyTorch.ipynb`](examples/getting_started_without_PyTorch.ipynb), which provides snippets about how to use Robust-Minisets data (the `.npz` files) without PyTorch.
* Simply change the super class of `Robust-Minisets` from `torch.utils.data.Dataset` to `collections.Sequence`, you will get a standard dataset without PyTorch. Check [`dataset_without_pytorch.py`](examples/dataset_without_pytorch.py) for more details.
* You still have most functionality of our Robust-Minisets code ;)

# Dataset

Please download the dataset(s) via Zenodo. You could also use our code to download automatically by setting `download=True` in [`dataset.py`](robust_minisets/dataset.py).

The Robust-Minisets collection contains several (mostly) test datasets. Each dataset (e.g., `tiny-imagenet-r.npz`) is comprised of up to 6 keys: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images` and `test_labels`.
* `train_images` / `val_images` / `test_images`: `N` × `W` × `H` × 3. `N` denotes the number of samples, `W` and `H` denote the width and height.  
* `train_labels` / `val_labels` / `test_labels`: `N` × `1`. `N` denotes the number of samples.

Following we provide a little overview on the datasets in `Robust-Minisets`:
* CIFAR-10.1
* CIFAR-10-C
* CIFAR-100-C
* EuroSAT
* EuroSAT-C
* MedMNIST-C
    * BreastMNIST-C
    * BloodMNIST-C
    * DermaMNIST-C
    * OCTMNIST-C
    * OrganAMNIST-C
    * OrganCMNIST-C
    * OrganSMNIST-C
    * PathMNIST-C
    * PneumoniaMNIST-C
    * TissueMNIST-C
* Tiny ImageNet
* Tiny ImageNet-A
* Tiny ImageNet-C
* Tiny ImageNet-R
* Tiny ImageNetv2

[Here](datasets.md) we provide a detailed summary to all datasets of the `Robust-Minisets` collection.

## Corruption Details
In this section we provide details about the structure of the corrupted (-C) datasets in the `Robust-Minisets`collection. In case you are interested in a detailed evaluation per corruption and/or severity level, the images in the datasets follow the same structure:
* Each dataset is of shape `N` $\cdot$ `C` $\cdot$ `S` × `W` × `H` × 3, where `N` denotes the number of test samples, `C` denotes the number of corruptions, and `S` denotes the number of severity levels (S=5).
* The images are ordered corruption by corruption and for each corruption from severity level 1 to 5
* The order of corruptions for each dataset and split can be found [here](datasets.md) or via the info attribute of each dataset (e.g. `TinyImageNetR.info["corruption_dict"]`)

# Command Line Tools

* List all available datasets:
    
        python -m robust_minisets available

* Download all available datasets:
    
        python -m robust_minisets download

* Delete all downloaded npz from root:

        python -m robust_minisets clean

* Print the dataset details given a dataset flag:

        python -m robust_minisets info --flag=<dataset_flag>

* Save the dataset as standard figure and csv files, which could be used for AutoML tools, e.g., Google AutoML Vision:

        python -m robust_minisets save --flag=<dataset_flag> --folder=tmp/ --postfix=png --download=True

    By default, `download=False`.

# License

The code is under [Apache-2.0 License](./LICENSE).

The publication licenses of the datasets can be found within the info dictionary via `robust_minisets.INFO[<dataset_flag>]`.

## Acknowledgements
This repository is built using the [timm](https://timm.fast.ai/) library and the [tiny-transformers](https://github.com/lkhl/tiny-transformers) repository.

This research was partly funded by Albert and Anneliese Konanz Foundation, the German Research Foundation under grant INST874/9-1 and the Federal Ministry of Education and Research Germany in the project M2Aind-DeepLearning (13FH8I08IA).

# Citing

If you find this work useful, please consider citing us:
```bibtex
@inproceedings{oehri2024genformer,
    title = {GenFormer – Generated Images are All You Need to Improve Robustness of Transformers on Small Datasets},
    author = {Oehri, Sven and Ebert, Nikolas and Abdullah, Ahmed and Stricker, Didier and Wasenm{\"u}ller, Oliver},
    booktitle = {International Conference on Pattern Recognition (ICPR)},
    year = {2024},
}
```

`DISCLAIMER`: Robust-Minisets is based on a wide range of existing datasets and benchmarks. Thus, please also cite source data paper(s) of the Robust-Miniset subset(s):
* [CIFAR-10.1](https://arxiv.org/abs/1806.00451)
* [EuroSAT](https://arxiv.org/abs/1709.00029)
* [ImageNet-A](https://arxiv.org/abs/1907.07174)
* [ImageNet-C](https://arxiv.org/abs/1903.12261)
* [ImageNet-R](https://arxiv.org/abs/2006.16241)
* [ImageNetv2](https://arxiv.org/abs/1902.10811)
* [MedMNIST](https://www.nature.com/articles/s41597-022-01721-8), the respective source datasets (described [here](https://medmnist.com))

# Release versions

* `v0.1.0`: Robust-Minisets beta release.
