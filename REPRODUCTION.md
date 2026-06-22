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

## Extension (in progress): cross-dataset transfer LEVIR → WHU-CD

To probe generalization, the LEVIR-trained model (best checkpoint, epoch 194) is evaluated
**zero-shot** (no retraining) on the WHU-CD test set. WHU-CD is a different building change-detection
dataset (Christchurch, NZ, aerial), preprocessed to 256×256 tiles.

| Metric (change class) | LEVIR (in-domain) | WHU (zero-shot) | Drop |
|---|---|---|---|
| F1 | 0.9027 | 0.7042 | −19.9 |
| IoU | 0.8226 | 0.5434 | −27.9 |
| Precision | 0.9247 | 0.6921 | −23.3 |
| Recall | 0.8817 | 0.7167 | −16.5 |

The same model drops ~20 F1 points moving to a new dataset with no retraining — a clear
cross-dataset generalization gap. Notably, **precision collapses more than recall** (−23.3 vs −16.5):
out-of-domain the model *over-flags* change (more false positives), the opposite of its
precision-heavy behaviour in-domain.

> Eval command: copy `best_ckpt.pt` into `checkpoints/whu_eval/`, then
> `python eval_cd.py --project_name whu_eval --data_name WHU --net_G base_transformer_pos_s4_dd8 --split test --gpu_ids 0`
> (passing `--net_G` is required; the default is a different architecture).

## How to reproduce

1. Clone this fork and set up the environment (PyTorch 2.x, torchvision, numpy).
2. Download LEVIR-CD and arrange it as the original README describes (`A/`, `B/`, `label/`, `list/`).
3. Train with the BIT config (`base_transformer_pos_s4_dd8`, settings above) — see `main_cd.py` / `scripts/`.
4. Evaluate the best checkpoint with `eval_cd.py`.

> **Trained weights** are not committed here (large binary). [link your Drive checkpoint here]

See the original [`README.md`](README.md) for full usage details from the authors.
