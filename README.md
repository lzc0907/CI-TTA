# SP-TTA: Shape‑Preserving Test‑Time Augmentation (for Domain Generalization)

> **TL;DR**：在推理阶段对测试样本做**形状保持**的弹性/网格微形变，并用**置信度过滤 + 软投票**融合；若过滤后“无票”，自动**回退到原图预测**。核心实现见 `tps.py`，评测脚本 `eval_tta_single.py`，一键脚本 `tta.sh`。

English summary: SP‑TTA builds several **shape‑preserving** TTA views (elastic and grid distortions + optional flip), then fuses predictions with **confidence filtering + soft voting**. If all views are filtered out, the final decision **falls back to the original image’s prediction**.


---

## ✨ 特性 | Features

- **形状保持的 TTA 视图**
  - 弹性形变 `elastic_deform`、网格扭曲 `grid_distortion`
  - `build_tta_views(x, ...)` 支持批量构造，始终**保留原图**作为 `views[0]`

- **多策略融合**
  - **Soft 平均**：`tta_predict_softmax`（等权平均概率）
  - **SP‑TTA（推荐）**：`tta_predict_conf`（置信度过滤 + 软投票；若过滤后为空集则**回退原图预测**）
 

- **即插即用评测**
  - `eval_tta_single.py`：加载已训模型 → 构造 TTA 视图 → 统计原图 / Soft 平均 / SP‑TTA 三者准确率
  - `tta.sh`：典型参数与命令行范式示例


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


## ⚙️ 超参建议 | Hyperparameters

- **弹性形变**：`alpha_std` 控幅度，`sigma` 控平滑（形变连续）。常用：`alpha_std=0.005~0.02`
- **网格扭曲**：`grid_rows/cols` 控网格密度，`distort_std` 控扭曲幅度；建议幅度**小**以不破坏语义形状
- **过滤阈值**：`conf_thres` 常在 `0.6~0.8`；阈值越高，保守性提升，**空集回退**更常出现


---

## 📊 复现实验 | Repro Tips

- `tta.sh` 展示了数据集（Office‑Home / PACS）、骨干网络（ResNet‑18/50）、算法（ERM/DANN/MMD/VREx 等）的典型参数组合
- 若你沿用 DeepDG 的训练脚本，可直接在其输出目录中选取 `best_model.pkl` 等权重放入评测脚本


---

## 🧩 常见问题 | FAQ

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

- 方法思路与实现参考了常见 DG 设定；仓库结构可与 DeepDG 工具链配合使用。
