# CI‑TTA: Class‑Invariant Test‑Time Augmentation for Domain Generalization

> **TL;DR**: At inference, build **shape‑preserving** elastic/grid deformations of each test sample and fuse predictions with **confidence filtering + soft voting**. If all augmented views are filtered out, **fall back to the original image’s prediction**. Core code in `tps.py`; evaluation via `eval_tta_single.py`; one‑click script in `tta.sh`.

---

## ✨ Features

- **Shape‑preserving TTA views**
  - Elastic deformation `elastic_deform`, grid distortion `grid_distortion`
  - `build_tta_views(x, …)` constructs views in batch and always **keeps the original** as `views[0]`

- **Flexible fusion strategies**
  - **Soft average**: `tta_predict_softmax` (uniform averaging over class probabilities)
  - **CI‑TTA (recommended)**: `tta_predict_conf` (confidence filtering + soft voting; if the post‑filter set is empty, **falls back to the original prediction**)

- **Plug‑and‑play evaluation**
  - `eval_tta_single.py`: load a trained model → build TTA views → report accuracy for *Origin* / *Soft* / *Origin‑CI‑TTA*
  - `tta.sh`: ready‑to‑edit example with typical arguments and command‑line usage

---

## 🔧 Installation

This repo is based on PyTorch. If you already use the DeepDG ecosystem, you can reuse the same Python environment and dependencies.

```bash
# Minimal example (choose versions as appropriate)
pip install torch torchvision
pip install numpy pillow opencv-python matplotlib pandas
```

---

## 🚀 Quickstart

### 1) Prepare data and a trained model
- **Datasets**: Office‑Home / PACS, organized as in DeepDG
- **Model**: any model you have trained (e.g., ERM‑ResNet‑50)

> `eval_tta_single.py` shares CLI parsing with your `train.py` (`get_args()`). To see all CLI options, check `train.py --help` in your project.

### 2) Option A: One‑click script (recommended—edit `tta.sh` to set paths/GPU first)
```bash
bash tta.sh
```

### 3) Option B: Run the Python script directly
```bash
python eval_tta_single.py \
  --data_dir <YOUR_DATA_DIR> \
  --dataset office-home \
  --net resnet50 \
  --algorithm ERM \
  --test_envs 2 \
  --max_epoch 120 \
  --lr 0.001 \
  --gpu_id 0
```

The script prints a side‑by‑side comparison:
- **Origin** (no TTA)
- **Soft** (uniform average over TTA views)
- **Origin‑CI‑TTA** (original + confidence filtering with fallback)
so you can quickly assess CI‑TTA’s benefit.

---

## ⚙️ Hyperparameters

- **Elastic deformation**: `alpha_std` (magnitude) and `sigma` (smoothness). Typical range: `alpha_std = 0.005–0.02`.
- **Grid distortion**: `grid_rows/cols` (grid density) and `distort_std` (magnitude). Prefer **small** distortions to preserve semantic shape.
- **Confidence threshold**: `conf_thres` typically `0.6–0.8`. Higher thresholds are more conservative, so **empty‑set fallback** occurs more often.

---

## 📊 Reproduction Tips

- `tta.sh` includes typical combinations for datasets (Office‑Home / PACS), backbones (ResNet‑18/50), and algorithms (ERM / DANN / MMD / VREx, etc.).
- If you reuse DeepDG training scripts, you can take the resulting weights (e.g., `best_model.pkl`) and point the evaluation script to them.

---

## 🧩 FAQ

- **Can I change inference without touching training?**  
  Yes. CI‑TTA only affects the **inference stage**. Load your trained weights, build TTA views, and fuse—no retraining required.

- **What if confidence filtering discards all views?**  
  CI‑TTA **falls back to the original image’s prediction** by design.

- **Where is the core implementation?**  
  Deformations and TTA utilities live in `tps.py`; end‑to‑end evaluation is in `eval_tta_single.py`; `tta.sh` shows common usage.

---

## 📚 Citation

If this repository or the implementation of CI‑TTA helps your research or product, please consider citing:

```bibtex
@inproceedings{CI-TTA,
  title     = {Class-Invariant Test-Time Augmentation for Domain Generalization},
  author    = {Zhicheng Lin and Xiaolin Wu and Xi Zhang},
  year      = {2025}
}
```

---

## 🤝 Acknowledgements

- The methodology follows common DG settings and is designed to work smoothly with the DeepDG toolchain and project structure.
