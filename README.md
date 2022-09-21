# Multi-Task Learning for dense prediction tasks (DenseMTL)

This repository provides the official source code and models for our [Multi-Task Learning for dense prediction tasks](https://arxiv.org/abs/2206.08927) paper (WACV 2023). The implementation is done using the [PyTorch](https://github.com/pytorch/pytorch) library.

<p align="center">
  <img src="./docs/dark.png#gh-dark-mode-only" width="700"/>
  <img src="./docs/light.png#gh-light-mode-only" width="700"/>
</p>

**DenseMTL: Multi-Task Learning for dense prediction tasks**\
[Ivan Lopes<sup>1</sup>](https://wonjunior.github.io/),
[Tuan-Hung Vu<sup>1,2</sup>](https://tuanhungvu.github.io/),
[Raoul de Charette<sup>1</sup>](https://team.inria.fr/rits/membres/raoul-de-charette/)</br>
<sup>1</sup> Inria, Paris, France.
<sup>2</sup> Valeo.ai, Paris, France.<br>

To cite our paper, please use:
```
@inproceedings{lopes2023densemtl,
  title={Cross-task Attention Mechanism for Dense Multi-task Learning},
  author={Lopes, Ivan and Vu, Tuan-Hung and de Charette, Raoul},
  booktitle={WACV},
  year={2023}
}
```


# Table of content

- [Multi-Task Learning for dense prediction tasks (DenseMTL)](#multi-task-learning-for-dense-prediction-tasks-densemtl)
- [Table of content](#table-of-content)
- [Overview](#overview)
- [Installation](#installation)
  - [1. Dependencies](#1-dependencies)
  - [2. Datasets](#2-datasets)
  - [3. Environment variables](#3-environment-variables)
- [Running DenseMTL](#running-densemtl)
  - [1. Command Line Interface](#1-command-line-interface)
  - [2. Experiments](#2-experiments)
  - [3. Models](#3-models)
  - [4. Evaluation](#4-evaluation)
  - [5. Visualization & Logging](#5-visualization--logging)
- [Project structure](#project-structure)
- [Credit](#credit)
- [License](#license)



# Overview

DenseMTL is an cross-attention based multi-task architecture which leverages multiple attention mechanisms to extract and enrich task features. As seen in the figure above, xTAM modules each receive a pair of differing task features to better assess cross task interactions and allow for an efficient cross talk distillation.

In total, this work covers a wide range of experiments, we summarize it by:

- ***3 settings***: fully-supervised (`FS`), semi-supervised auxiliary depth (`SDE`), and domain adaptation (`DA`).
- ***4 datasets***: Cityscapes, Virtual Kitti 2, Synthia, and NYU-Depth v2.
- ***4 tasks***: semantic segmentation (`S`), depth regression (`D`), surface normals estimation (`N`), and edge detection (`E`).
- ***3 task sets***: `{S, D}`, `{S, D, N}`, and `{S, D, N, E}`.


# Installation

## 1. Dependencies
First create a new conda environment with the required packages found in `environment.yml`. You can do so with the following line:
```
>>> conda env create -n densemtl -f environment.yml
```

Then activate environment `densemtl` using:
```
>>> conda activate densemtl
```

## 2. Datasets

* **CITYSCAPES**: Follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/)
  to download the images and validation ground-truths. Please follow the dataset directory structure:
  ```html
  <CITYSCAPES_DIR>/             % Cityscapes dataset root
  ├── leftImg8bit/              % input image (leftImg8bit_trainvaltest.zip)
  ├── leftImg8bit_sequence/     % sequences need for monodepth (disparity_sequence_trainvaltest.zip)
  ├── disparity/                % stereo depth (disparity_trainvaltest.zip)
  ├── camera/                   % camera parameters (camera_trainvaltest.zip)
  └── gtFine/                   % semantic segmentation labels (gtFine_trainvaltest.zip)
  ```


* **SYNTHIA**: Follow the instructions [here](http://synthia-dataset.net/downloads/) to download the images from the *SYNTHIA-RAND-CITYSCAPES (CVPR16)* split. Download the segmentation labels from [CTRL-UDA](https://github.com/susaha/ctrl-uda/blob/main/README.md) using the link  [here](https://drive.google.com/file/d/1TA0FR-TRPibhztJI5-OFP4iBNaDDkQFa/view?usp=sharing).
  Please follow the dataset directory structure:
  ```html
  <SYNTHIA_DIR>/                % Synthia dataset root
  ├── RGB/                      % input images
  ├── parsed_LABELS/            % semseg labels labels
  └── Depth/                    % depth labels
  ```

* **VKITTI2**: Follow the instructions [here](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) to download the images from the Virtual KITTI 2 dataset. Please follow the dataset directory structure:
  ```html
  <VKITTI2_DIR>/                % VKITTI 2 dataset root
  ├── rgb/                      % input images (vkitti_2.0.3_rgb.tar)
  ├── classSegmentation/        % semseg labels (vkitti_2.0.3_classSegmentation.tar)
  └── depth/                    % depth labels (vkitti_2.0.3_depth.tar)
  ```

* **NYUDv2**: Follow the instructions [here](https://github.com/brdav/atrc) to download the NYUDv2 dataset along with its semantic segmentation, normals, depth, and edge labels. Please follow the dataset directory structure:
  ```html
  <NYUDV2_DIR>/                 % NYUDv2 dataset root
  ├── images/                   % input images
  ├── segmentation/             % semseg labels
  ├── depth/                    % depth labels
  ├── edge/                     % semseg labels
  └── normals/                  % normals labels
  ```

## 3. Environment variables

Update `configs/env_config.yml` with the path to the different directories by defining:
- Saved models path: `MODELS`.
- Datasets paths `CITYSCAPES_DIR`, `SYNTHIA_DIR`, `VKITTI2_DIR`, and `NYUDV2_DIR`.
- Logs path: `LOG_DIR`.

All constants provided in this file are loaded as environment variables and accessible at runtime via `os.environ`. Alternatively those constants can be defined in the command line before running the project.


# Running DenseMTL

## 1. Command Line Interface
The following are the command line inferface arguments and options:
```
--env-config ENV_CONFIG
  Path to file containing the environment paths, defaults to configs/env_config.yml.
--base CONFIG
  Optional path to base configuration yaml file, can be left unused if --config file contains all keys
--config CONFIG
  Path to main configuration yaml file.
--project PROJECT
  Project name for logging and used as wandb project
--resume
  Flag to resume training, this will look for last available model checkpoint from the same setup
--evaluate PATH
  Will load the model provided at the file path and perform evaluation using the config setup.
-s SEED, --seed SEED
  Seed for training and dataset.
-d, --debug
  Flag to perform single validation inference for debugging purposes.
-w, --disable-wandb
  Flag to disable Weight & Biases logging
```

Experiments are based off of configuration files. Overall each configuration file must follow this structure:

```yaml
setup:
  name: exp-name
  model:
    └── model args
  loss:
    └── loss args
  lr:
    └── learning rates
data:
  └── data module args
training:
  └── training args
optim:
  └── optimizer args
scheduler:
  └── scheduler args
```

For arguments which are recurring across experiments such as `data`, `training`,  `optim`, `scheduler`, we use a base configuration file that we pass to the process via the `--base` option. The two configuration files (provided with `--base` and `--config`) are merged together at the top level (`config` can overwrite `base`). See more details in [main.py](main.py).

Environment variables can be referenced inside the configuration file by using the `$ENV:` prefix, *eg.*: `path: $ENV:CITYSCAPES_DIR`.

## 2. Experiments

To reproduce the experiments, you can run the following scripts.

- Single task learning baseline:
  ```
  python main.py \
    --base=configs/<dataset>/fs_bs2.yml \
    --config=configs/<dataset>/resnet101_STL_<task>.yml
  ```
  Where `<dataset>` $\in$ {`cityscapes`, `synthia`, `vkitti2`, `nyudv2`} and `<task>` $\in$ {`S`, `D`, `N`, `E`}.


- Our method on the *fully supervised* setting (FS):
  ```
  python main.py \
    --base=configs/<dataset>/fs_bs2.yml \
    --config=configs/<dataset>/resnet101_ours_<set>.yml
  ```
  Where `<dataset>` $\in$ {`cityscapes`, `synthia`, `vkitti2`, `nyudv2`} and `<set>` $\in$ {`SD`, `SDN`, `SDNE`}.

  Do note, however, that experiments with the edge estimation task are only performed on the NYUDv2 dataset.

- Our method on *semi-supervised depth estimation* (SDE):
  ```
  python main.py --config configs/cityscapes/monodepth/resnet101_ours_SD.yml
  ```

- Our method on *domain adaptation* (DA):
  ```
  python main.py \
    --base=configs/da/<dataset>/fs_bs2.yml \
    --config configs/da/<dataset>/resnet101_ours_SD.yml
  ```
  Where `<dataset>` $\in$ {`sy2cs` (for Synthia $\mapsto$ Cityscapes), `vk2cs` (for VKITTI2 $\mapsto$ Cityscapes)}.

## 3. Models

*The weights of our models will be released soon.*


## 4. Evaluation

To evaluate a model, the `--evaluate` option can be set with a path to the state dictionnary `.pkl` file. This weight file will be loaded onto the model and the evaluation loop launched. Keep in mind you also need to provide a valid configuration files in order to evaluate our method with weights located in `weights/vkitti2_densemtl_SD.pkl`, simply run `python main.py --config=configs/vkitti2/resnet101_ours_SD.yml --evaluate=weights/vkitti2densemtl.pkl`



## 5. Visualization & Logging

By default, visualizations, losses, and metrics are logged using [Weights & Biases](https://wandb.ai). In case you do not wish to log your trainings and evaluations through this tool, you can disable it by using the `--disable-wandb` flag. In all cases, the loss values and metrics are logged via the standard output.

Checkpoints, models and configuration files are saved under the `LOG_DIR` directory folder. More specifically, those will be located under `<LOG_DIR>/<dataset>/<config-name>/s<seed>/<timestamp>`. For example you could have something like: `<LOG_DIR>/vkitti2/resnet101_ours_SD/s42/2022-04-19_10-09-49` for a `SD` training of our method on VKITTI2 with a seed equal to `42`.


# Project structure

The [`main.py`](main.py) file is the entry point to perform training and evaluation on the different setups.

```
root
  ├── configs/    % Configuration files to run the experiments
  ├── training/   % Training loops for all settings
  ├── dataset/    % PyTorch dataset definitions as well as semantic segmentation encoding logic
  ├── models/     % Neural network modules, inference logic and method implementation
  ├── optim/      % Optimizers related code
  ├── loss/       % Loss modules for each task type includes the metric and visualization calls
  ├── metrics/    % Task metric implementations
  ├── vendor/     % Third party source code
  └── utils/      % Utility code for other parts of the code
```

# Credit
This repository contains code taken from [valeoai's ADVENT](https://github.com/valeoai/ADVENT), [SimonVandenhende's MTL-survey](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch), [nianticlabs's Monodepth2](https://github.com/nianticlabs/monodepth2), and [lhoyer's 3-Ways](https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth).

# License
DenseMTL is released under the [Apache 2.0 license](./LICENSE).

---

[↑ back to top](#multi-task-learning-for-dense-prediction-tasks-densemtl)
