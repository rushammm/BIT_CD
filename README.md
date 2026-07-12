# BIT-CD: reproduction + foundation-model backbone study

> **Fork note.** This is a fork of [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD), the official implementation of *"Remote Sensing Image Change Detection with Transformers"* (Chen, Qi & Shi, IEEE TGRS 2021). **All credit for the method and original code belongs to the authors.** Everything in this section is my own reproduction and extension; the authors' original README follows below.

I reproduced BIT-CD on LEVIR-CD, then asked whether a foundation-model backbone would help it generalize to a *different* change-detection dataset without retraining.

**Reproduction.** Trained BIT on LEVIR-CD and matched the published result (F1 0.9027 vs the paper's 0.8931), after three one-line compatibility fixes for PyTorch 2.x / modern numpy. Details in [`reproduction.md`](reproduction.md).

**The problem.** That same LEVIR-trained model, run zero-shot on WHU-CD, drops ~20 F1 points (0.903 → 0.704). Precision falls further than recall, so out-of-domain it *over-flags* change.

**The experiment.** I swapped BIT's ImageNet ResNet backbone for two foundation models, frozen and fine-tuned, with a small adapter reshaping ViT patch tokens into BIT's spatial feature map: **DOFA** (Earth-observation pretrained) and **DINOv2** (general-purpose vision).

| Backbone | LEVIR F1 | WHU F1 (zero-shot) | Keeps |
|---|---:|---:|---:|
| ResNet (BIT baseline) | 0.903 | 0.704 | 78% |
| DOFA frozen | 0.721 | 0.693 | 96% |
| DOFA fine-tuned | 0.724 | 0.518 | 72% |
| **DINOv2 frozen** | 0.773 | **0.781** | **101%** |
| DINOv2 fine-tuned | 0.803 | 0.386 | 48% |

**Finding.** My hypothesis was that the EO-specific model would transfer better. It did not. The axis that mattered was **frozen vs fine-tuned**, not EO vs general: frozen features held up across the domain shift while fine-tuned ones collapsed (fine-tuned DINOv2's precision fell to 0.266, over-flagging unchanged buildings), and general-purpose DINOv2 beat EO-specific DOFA.

**Caveats.** Single seed, one dataset pair, one transfer direction, and a deliberately simple adapter that caps in-domain F1. This is research-grade evidence, not a benchmark. Full caveats and the failure-mode figures are in **[`bit-cd-foundation-results.md`](bit-cd-foundation-results.md)**; the code is in [`foundation_backbone_comparison.ipynb`](foundation_backbone_comparison.ipynb) and [`dofa_bit_integration.ipynb`](dofa_bit_integration.ipynb).

---

# Original README (Chen, Qi & Shi)

Here, we provide the pytorch implementation of the paper: Remote Sensing Image Change Detection with Transformers.

For more ore information, please see our published paper at [IEEE TGRS](https://ieeexplore.ieee.org/document/9491802) or [arxiv](https://arxiv.org/abs/2103.00208). 

![image-20210228153142126](./images/pipeline.png)

## Requirements

```
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
```

## Installation

Clone this repo:

```shell
git clone https://github.com/justchenhao/BIT_CD.git
cd BIT_CD
```

## Quick Start

We have some samples from the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) dataset in the folder `samples` for a quick start.

Firstly, you can download our BIT pretrained model——by [baidu drive, code: 2lyz](https://pan.baidu.com/s/1HiXwpspl6odYQKda6pMuZQ) or [google drive](https://drive.google.com/file/d/1IVdF5a3e1_7DiSndtMkhpZuCSgDLLFcg/view?usp=sharing). After downloaded the pretrained model, you can put it in `checkpoints/BIT_LEVIR/`.

Then, run a demo to get started as follows:

```python
python demo.py 
```

After that, you can find the prediction results in `samples/predict`.

## Train

You can find the training script `run_cd.sh` in the folder `scripts`. You can run the script file by `sh scripts/run_cd.sh` in the command environment.

The detailed script file `run_cd.sh` is as follows:

```cmd
gpus=0
checkpoint_root=checkpoints 
data_name=LEVIR  # dataset name 

img_size=256
batch_size=8
lr=0.01
max_epochs=200  #training epochs
net_G=base_transformer_pos_s4_dd8 # model name
#base_resnet18
#base_transformer_pos_s4_dd8
#base_transformer_pos_s4_dd8_dedim8
lr_policy=linear

split=train  # training txt
split_val=val  #validation txt
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

## Evaluate

You can find the evaluation script `eval.sh` in the folder `scripts`. You can run the script file by `sh scripts/eval.sh` in the command environment.

The detailed script file `eval.sh` is as follows:

```cmd
gpus=0
data_name=LEVIR # dataset name
net_G=base_transformer_pos_s4_dd8_dedim8 # model name 
split=test # test.txt
project_name=BIT_LEVIR # the name of the subfolder in the checkpoints folder 
checkpoint_name=best_ckpt.pt # the name of evaluated model file 

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

DSIFN-CD: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Citation

If you use this code for your research, please cite our paper:

```
@Article{chen2021a,
    title={Remote Sensing Image Change Detection with Transformers},
    author={Hao Chen, Zipeng Qi and Zhenwei Shi},
    year={2021},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095166}
}
```

