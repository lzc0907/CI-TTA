# SP-TTA: Shape‑Preserving Test‑Time Augmentation (for Domain Generalization)

> **TL;DR**：在推理阶段对测试样本做**形状保持**的弹性/网格微形变（可选翻转），并用**置信度过滤 + 软投票**融合；若过滤后“无票”，自动**回退到原图预测**。核心实现见 `tps.py`，评测脚本 `eval_tta_single.py`，一键脚本 `tta.sh`。

English summary: SP‑TTA builds several **shape‑preserving** TTA views (elastic and grid distortions + optional flip), then fuses predictions with **confidence filtering + soft voting**. If all views are filtered out, the final decision **falls back to the original image’s prediction**.


---

## ✨ 特性 | Features

- **形状保持的 TTA 视图**
  - 弹性形变 `elastic_deform`、网格扭曲 `grid_distortion`、可选水平翻转 `hflip`
  - `build_tta_views(x, ...)` 支持批量构造，始终**保留原图**作为 `views[0]`

- **多策略融合**
  - **Soft 平均**：`tta_predict_softmax`（等权平均概率）
  - **SP‑TTA（推荐）**：`tta_predict_conf`（置信度过滤 + 软投票；若过滤后为空集则**回退原图预测**）
  - **硬投票（可选）**：`tta_predict_vote`（平局可回退软投票）

- **即插即用评测**
  - `eval_tta_single.py`：加载已训模型 → 构造 TTA 视图 → 统计原图 / Soft 平均 / SP‑TTA 三者准确率
  - `tta.sh`：典型参数与命令行范式示例


---

## 📁 目录结构 | Repo Layout

```
.
├── tps.py                  # TTA 视图与融合策略实现（elastic/grid/flip + 多种融合）
├── eval_tta_single.py      # 单模型评测示例（加载权重 + 视图构造 + 指标统计）
├── tta.sh                  # 示例运行脚本（按需改数据/模型路径/显卡ID等）
└── readme.md               # 原 DeepDG 仓库说明（如有）
```


---

## 🔧 环境 | Installation

本项目基于 PyTorch。若你来自 DeepDG 生态，直接沿用其 Python/依赖版本即可。

```bash
# 最小示例（请按需选择对应版本）
pip install torch torchvision
pip install numpy pillow opencv-python matplotlib pandas
```


---

## 🚀 快速开始 | Quickstart

### 1) 准备数据与模型
- 数据目录：建议使用 Office‑Home / PACS 等，目录组织与 DeepDG 一致
- 模型：使用你已训练好的模型权重（例如 ERM‑ResNet‑50）

> `eval_tta_single.py` 复用 `train.py` 的参数解析（`get_args()`）。如需查看所有 CLI 选项，请参考你项目中的 `train.py --help`。

### 2) 方式 A：一键脚本（推荐先打开 `tta.sh` 修改路径/显卡等）
```bash
bash tta.sh
```

### 3) 方式 B：直接运行 Python 脚本
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

脚本将打印：
- **Origin**（只用原图）
- **Soft**（TTA 视图等权平均）
- **Origin‑SP‑TTA**（原图 + 置信度过滤 + 回退）
三者的准确率对比，便于快速评估 SP‑TTA 的收益。


---

## 🧠 方法一览 | Method at a Glance

### 视图构造（Views）
```python
from tps import build_tta_views

views = build_tta_views(
    batch,                       # [B,C,H,W], 视图列表将以 batch 的原图为 views[0]
    num_views=100,               # 总视图数（包含原图）
    do_elastic=True,             # 是否启用弹性形变
    do_grid=True,                # 是否启用网格扭曲
    elastic_params=dict(alpha_std=0.005, sigma=10.0),
    grid_params=dict(grid_rows=3, grid_cols=3, distort_std=0.005),
    include_flip=True            # 是否加入水平翻转
)
```

### 融合策略（Fusion）

**Soft 平均**
```python
from tps import tta_predict_softmax

probs = tta_predict_softmax(model, batch, views)  # [B, num_classes]
preds = probs.argmax(dim=1)
```

**SP‑TTA（置信度过滤 + 空集回退原图）**
```python
from tps import tta_predict_conf

final_preds, records = tta_predict_conf(
    model, batch, views, labels,
    conf_thres=0.7,              # 置信度阈值（越高越保守）
    return_record=True,
    batch_id=0
)
# 备注：当某样本在过滤后完全没有有效投票时，函数会回退到原图预测；
# 你也可以在 records 中统计 fallback 情况（若实现中开启了该标记）。
```

**硬投票（可选）**
```python
from tps import tta_predict_vote
preds = tta_predict_vote(model, batch, views, conf_thres=0.7)
```


---

## ⚙️ 超参建议 | Hyperparameters

- **弹性形变**：`alpha_std` 控幅度，`sigma` 控平滑（形变连续）。常用：`alpha_std=0.005~0.02, sigma=8~12`
- **网格扭曲**：`grid_rows/cols` 控网格密度，`distort_std` 控扭曲幅度；建议幅度**小**以不破坏语义形状
- **过滤阈值**：`conf_thres` 常在 `0.6~0.8`；阈值越高，保守性提升，**空集回退**更常出现


---

## 📊 复现实验 | Repro Tips

- `tta.sh` 展示了数据集（Office‑Home / PACS）、骨干网络（ResNet‑18/50）、算法（ERM/DANN/MMD/VREx 等）的典型参数组合
- 若你沿用 DeepDG 的训练脚本，可直接在其输出目录中选取 `best_model.pkl` 等权重放入评测脚本


---

## 🧩 常见问题 | FAQ

- **报错 “Dimension out of range … got 1”**  
  大多因为 `model.predict` 在 `B==1` 时 squeeze 成 `[num_classes]`。请确保模型输出被 reshape 为 `[B, num_classes]`；本仓库的 `tta_predict_conf` 等函数已对单样本 / squeeze 做了鲁棒处理。

- **如何只改推理，不动训练？**  
  SP‑TTA 只影响**推理阶段**。你可以直接加载已有权重，构造 TTA 视图并融合，无需重新训练。


---

## 📚 引用 | Citation

如果本仓库或 SP‑TTA 的实现对你的研究或产品有帮助，请引用本文工作（示例）：

```bibtex
@inproceedings{your_sp_tta_year,
  title     = {SP-TTA: Shape-Preserving Test-Time Augmentation for Domain Generalization},
  author    = {Your Name and ...},
  booktitle = {...},
  year      = {2025}
}
```

---

## 🤝 致谢 | Acknowledgements

- 方法思路与实现参考了常见 DG/TTA 设定；仓库结构可与 DeepDG 工具链配合使用。
