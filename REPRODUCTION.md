# Reproduction — BIT-CD on LEVIR-CD

> This is a **fork** of [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD),
> the official code for *"Remote Sensing Image Change Detection with Transformers"*
> (Chen, Qi, Shi — IEEE TGRS 2021). All credit for the method and original code is theirs.
>
> **My contribution** is reproducing their LEVIR-CD result and documenting it here:
> minimal compatibility fixes to run the 2021 code on modern PyTorch, plus the trained
> result below.

## What I changed

The original code targets older library versions. Three one-line fixes were needed to run
it on PyTorch 2.x / current numpy (see commit `compat fixes for PyTorch 2.x / numpy`):

| File | Change | Why |
|---|---|---|
| `datasets/CD_dataset.py` | `np.str` → `str` | `np.str` was removed in newer numpy |
| `models/basic_model.py` | added `weights_only=False` to `torch.load` | PyTorch 2.6 changed the default to `True` |
| `models/resnet.py` | import `load_state_dict_from_url` from `torch.hub` | `torchvision.models.utils` was removed |

No changes to the model, loss, or training logic — the goal was to reproduce, not modify.

## Setup

- **Dataset:** LEVIR-CD
- **Model:** `base_transformer_pos_s4_dd8` (BIT)
- **Training:** batch size 8, SGD, lr 0.01 (linear decay), 200 epochs, `ce` loss, 256×256 images
- **Hardware:** single GPU (trained on Google Colab)

## Results (LEVIR-CD test set)

Metrics are for the **change class** unless noted. Best checkpoint (epoch 194) evaluated on the test split.

| Metric | This reproduction | Paper (BIT) |
|---|---|---|
| F1 (change) | **0.9027** | 0.8931 |
| IoU (change) | **0.8226** | 0.8068 |
| Overall Accuracy | **0.9903** | 0.9892 |
| Precision (change) | 0.9247 | 0.8924 |
| Recall (change) | 0.8817 | 0.8937 |

The reproduction matches and slightly exceeds the reported F1. One difference worth noting:
this run is **precision-heavy / recall-light** relative to the paper's balanced precision/recall —
i.e. it is slightly more conservative about flagging change. Net F1 still comes out ahead.

## How to reproduce

1. Clone this fork and set up the environment (PyTorch 2.x, torchvision, numpy).
2. Download LEVIR-CD and arrange it as the original README describes (`A/`, `B/`, `label/`, `list/`).
3. Train with the BIT config (`base_transformer_pos_s4_dd8`, settings above) — see `main_cd.py` / `scripts/`.
4. Evaluate the best checkpoint with `eval_cd.py`.

> **Trained weights** are not committed here (large binary). [link your Drive checkpoint here]

See the original [`README.md`](README.md) for full usage details from the authors.
