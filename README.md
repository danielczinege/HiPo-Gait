# HiPo-Gait: Hierarchical Pose model decoupling for Gait Recognition


[![https://doi.org/10.1109/TBIOM.2026.3684931](https://img.shields.io/badge/doi-10.1109%2FTBIOM.2026.3684931-blue)](https://doi.org/10.1109/TBIOM.2026.3684931)
[![https://www.researchgate.net/publication/403942426_HiPo-Gait_Hierarchical_Pose_model_decoupling_for_Gait_Recognition](https://img.shields.io/badge/%20-Read_on_ResearchGate-blue.svg?logo=researchgate&color=00CCBB&labelColor=555555)](https://www.researchgate.net/publication/403942426_HiPo-Gait_Hierarchical_Pose_model_decoupling_for_Gait_Recognition)

Source code for the manuscript **HiPo-Gait** presented in the publication for *IEEE Transactions on Biometrics, Behavior, and Identity Science (IEEE T-BIOM)*.

Some parts of code are derived from [OpenGait](https://github.com/ShiqiYu/OpenGait).

## Abstract

Gait recognition has traditionally relied on silhouette- or appearance-based representations to analyze walking patterns. While silhouettes capture rich movement information, they are highly dependent on body shape and contours, which are potentially irrelevant to gait analysis. The human pose emerges as a more robust and semantically meaningful alternative. However, pose-based models have typically underperformed compared to silhouette-based approaches. To enhance gait feature extraction from pose representations, this paper proposes HiPo-Gait, a novel hierarchical architecture that analyzes isolated limbs of the body pose and groups them at different levels using a new local attention-based fusion module (LAF) to produce a robust hierarchical descriptor of the walking pattern. Moreover, instead of using point coordinates as input, we use heatmaps, a richer representation of the body pose. 
Our experimental results indicate that *a)* the hierarchical decoupling extracts richer features from every individual limb and *b)* the fusion approach optimally aggregates the limbs, outperforming the classical fusion operations. Finally, our approach achieves a percentage point increase of 3.3% over the top pose-based state-of-the-art model in CASIA-B, 21.9%, and 25.6%, respectively, in CCPG and SUSTech1K, and 1.7% improvement over the top model in the FVG-B dataset.

## Installation

Clone this repo, create a virtual environment (if you prefer), and install the requirements through **pip**:

```bash
pip install -r requirements.txt
```
## Data pretreatment

Pose heatmaps are estimated from RGB using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose). Please refer to the original repo to reproduce computation. 

To preprocess the raw heatmaps into the format accepted by our script, follow the instructions at [heatmap pretreatment](./docs/heatmap_pretreat.md/).

## Prepare *cfg* files

Prepare your cfg file from the templates in [model_cfgs](model_cfgs/). **Make sure to set your preprocessed pose directory in `data_cfg.dataset_root`.**

## Train & test

Train or test a model by running: 

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
                         --nproc_per_node=2 \
                         src/train_test.py \
                         --cfgs <your cfg file> \
                         --phase <train for training or test for testing>
```
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train` or `test`.
- `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there.
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously.
- `--out_dir` Path to output directory. By default, output checkpoints and logs will be placed at *output/* directory.

**Note:** Multinode training is not supported yet.

**Checkpoints**: You can evaluate also our trained models by downloading the model checkpoints found in [checkpoints](checkpoints/) dir.
Download the model weights and set their path in the `evaluator_cfg.restore_hint` parameter.


## Citation

If our work is useful for you, please consider to cite us using this *bibtex* citation:

```
@ARTICLE{cubero2026hipogait,
  author={Cubero, Nicolas and Castro, Francisco M. and Guil, Nicolas and Marin-Jimenez, Manuel J.},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={HiPo-Gait: Hierarchical Pose model decoupling for Gait Recognition}, 
  year={2026},
  pages={1-1},
  keywords={Feeds;Antennas;Frequency modulation;Radio broadcasting;Frequency modulation;Videos;Modulation;Radio broadcasting;Video equipment;Protocols;Gait recognition;human pose;deep learning;hierarchical pose decoupling},
  doi={10.1109/TBIOM.2026.3684931}
}
```
