# coding=utf-8

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

# ---------- 基础工具：生成高斯核 ----------
def _gaussian_kernel2d(kernel_size=21, sigma=3.0, device='cpu'):
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)

def _smooth_displacement(disp, sigma=8.0):
    """对位移场做高斯平滑；disp: [B,2,H,W]"""
    B, C, H, W = disp.shape
    ksize = int(2 * round(3*sigma) + 1)  # ~6*sigma 覆盖
    k = _gaussian_kernel2d(ksize, sigma, device=disp.device)
    # 对 x,y 两个通道分别卷积
    disp_x = F.conv2d(disp[:, 0:1], k, padding=ksize//2)
    disp_y = F.conv2d(disp[:, 1:2], k, padding=ksize//2)
    return torch.cat([disp_x, disp_y], dim=1)

def _make_base_grid(B, H, W, device):
    """归一化 base grid [-1,1]"""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    base = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
    base = base.unsqueeze(0).repeat(B, 1, 1, 1)   # [B,H,W,2]
    return base

def _apply_flow(x, flow_xy):
    """
    x: [B,C,H,W]
    flow_xy: [B,2,H,W]，以归一化坐标（-1..1）位移
    """
    B, C, H, W = x.shape
    base = _make_base_grid(B, H, W, x.device)     # [B,H,W,2]
    grid = base + flow_xy.permute(0,2,3,1)        # [B,H,W,2]
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)



def elastic_deform(x, alpha_mean=0, alpha_std=0.05, sigma=10.0): 
    """
    alpha_mean/std: 控制形变幅度的高斯分布参数
    sigma：高斯平滑的标准差（像素）
    """
    B, C, H, W = x.shape

    # 每张图采样一个 alpha（保证非负）
    alpha = torch.randn(B, device=x.device) * alpha_std + alpha_mean
    alpha = alpha.abs().view(B, 1, 1, 1)  # [B,1,1,1]

    # 像素位移场 ~ N(0,1)
    disp = torch.randn(B, 2, H, W, device=x.device)
    disp = _smooth_displacement(disp, sigma=sigma)

    # 归一化到 [-alpha, alpha]（每张图独立）
    disp = disp / (disp.abs().amax(dim=(2,3), keepdim=True) + 1e-8) * alpha

    return _apply_flow(x, disp)


def grid_distortion(x, grid_rows=4, grid_cols=4, distort_mean=0, distort_std=0.01): 
    """
    grid_rows/cols：控制点网格密度（越小块越大）
    distort_mean/std: 高斯分布参数，用于采样扭曲幅度
    """
    B, C, H, W = x.shape
    gh = grid_rows + 1
    gw = grid_cols + 1

    # distort 从高斯分布采样 (保证非负)
    distort = torch.randn(1).item() * distort_std + distort_mean
    distort = abs(distort)  

    # 随机控制点位移
    ctrl = (torch.rand(B, 2, gh, gw, device=x.device) * 2 - 1) * distort  
    dense = F.interpolate(ctrl, size=(H, W), mode='bicubic', align_corners=True)
    return _apply_flow(x, dense)

# ---------- 水平翻转 ----------
def hflip(x):
    return torch.flip(x, dims=[-1])




def build_tta_views(x, num_views=100,
                           do_elastic=True, do_grid=True,
                           elastic_params=dict(alpha_std=0.08, sigma=8.0),
                           grid_params=dict(grid_rows=4, grid_cols=4, distort_std=0.01),
                           include_flip=True):
    """
    随机生成 TTA 视图，每个视图随机应用 flip / elastic / grid 中 1~3 种变换。
    保证总共返回 num_views 张视图（包含原图）。
    """
    B, C, H, W = x.shape
    views = [x]  # 原图一定保留
    transform_funcs = []

    if do_elastic:
        transform_funcs.append(lambda img: elastic_deform(img, **elastic_params))
    if do_grid:
        transform_funcs.append(lambda img: grid_distortion(img, **grid_params))
    if include_flip:
        transform_funcs.append(hflip)

    while len(views) < num_views:
        v = x.clone()
        # 随机选择 1~len(transform_funcs) 个变换
        k = random.randint(1, len(transform_funcs))
        funcs = random.sample(transform_funcs, k)
        for f in funcs:
            v = f(v)
        views.append(v)

    # 如果生成多于 num_views，则裁剪
    return views[:num_views]





@torch.no_grad()
def tta_predict_softmax(model, batch, views):
    """
    对 batch 的若干 TTA 视图做前向，softmax 后平均。
    model.predict 输入形状：[B,C,H,W]，输出 logits [B,num_classes]
    """
    probs = []
    for v in views:
        logits = model.predict(v)
        probs.append(F.softmax(logits, dim=1))
    prob_mean = torch.stack(probs, dim=0).mean(dim=0)  # [B,C]
    return prob_mean




@torch.no_grad()
def tta_predict_vote(model, batch, views, conf_thres=0.55):
    """
    TTA 硬投票 + 置信度过滤 + 平局时使用平均概率软投票。
    
    参数：
      model.predict: 输入 [B,C,H,W] -> 输出 logits [B,num_classes]
      batch: 原始 batch [B,C,H,W]
      views: TTA 视图的列表，每个元素 [B,C,H,W]
      conf_thres: 投票的置信度阈值
    返回：
      final_preds: [B] 最终预测类别（平局时用软投票，没票时 -1）
    """
    B = batch.size(0)
    num_classes = None
    vote_counts = None  # [B, num_classes]
    prob_sums = None    # 用于平局时软投票

    for v in views:
        logits = model.predict(v)            # [B, num_classes]
        probs = torch.softmax(logits, dim=1) # [B, num_classes]
        preds = probs.argmax(dim=1)          # [B]
        confs = probs[torch.arange(B), preds]# [B]

        if num_classes is None:
            num_classes = logits.size(1)
            vote_counts = torch.zeros(B, num_classes, device=logits.device)
            prob_sums = torch.zeros(B, num_classes, device=logits.device)

        # 累积概率（用于平局时回退到 soft vote）
        prob_sums += probs

        # 只对高置信度结果投票
        mask = confs >= conf_thres
        if mask.any():
            idx = torch.arange(B, device=logits.device)
            vote_counts[idx[mask], preds[mask]] += 1

    # 硬投票结果
    final_preds = vote_counts.argmax(dim=1)  # [B]

    # 检测平局
    max_votes, _ = vote_counts.max(dim=1)    # [B]
    ties = (vote_counts == max_votes.unsqueeze(1)).sum(dim=1) > 1  # 平局 mask

    # 对平局的样本，使用平均概率的 argmax
    if ties.any():
        soft_preds = prob_sums[ties].argmax(dim=1)
        final_preds[ties] = soft_preds

    # 对完全没票的样本（全0），设置为 -1
    no_votes = (vote_counts.sum(dim=1) == 0)
    final_preds[no_votes] = -1

    return final_preds






@torch.no_grad()
def tta_predict_conf(
    model, batch, views, labels, 
    conf_thres=0, return_record=True, batch_id=0
): 
    """
    TTA 软投票 + 置信度过滤 + 扰动图 L2 距离（相对于 views[0]）
    若过滤后为空集，则回退到原图预测。
    """
    B = batch.size(0)

    # -------- views[0] 作为原图预测 --------
    logits0 = model.predict(views[0])
    probs0  = torch.softmax(logits0, dim=1)
    preds0  = probs0.argmax(dim=1)                     # <--- 保存原图预测以便回退
    confs0  = probs0[torch.arange(B), preds0]

    final_prob_sums = torch.zeros_like(probs0)

    records = []
    if return_record:
        for i in range(B):
            records.append({
                "batch_id": batch_id,
                "sample_id": i,
                "img_type": "origin",
                "conf": float(confs0[i]),
                "correct": int(preds0[i].item() == labels[i].item()),
                "perturb_dist": 0.0,
                "probs": probs0[i].cpu().tolist()
            })

    mask0 = confs0 >= conf_thres
    final_prob_sums[mask0] += probs0[mask0]

    # -------- 扰动图预测 --------
    first_view = views[0]
    for vi, v in enumerate(views[1:], start=1):
        logits_i = model.predict(v)
        probs_i  = torch.softmax(logits_i, dim=1)
        preds_i  = probs_i.argmax(dim=1)
        confs_i  = probs_i[torch.arange(B), preds_i]

        # 与原图的 L2 距离
        l2_dist = torch.norm(
            v.view(v.size(0), -1) - first_view.view(first_view.size(0), -1),
            dim=1
        )

        mask_i = confs_i >= conf_thres
        final_prob_sums[mask_i] += probs_i[mask_i]

        if return_record:
            for i in range(B):
                records.append({
                    "batch_id": batch_id,
                    "sample_id": i,
                    "img_type": f"view{vi}",
                    "conf": float(confs_i[i]),
                    "correct": int(preds_i[i].item() == labels[i].item()),
                    "perturb_dist": float(l2_dist[i]),
                    "probs": probs_i[i].cpu().tolist()
                })

    # -------- 最终预测（软投票 + 回退） --------
    no_votes   = (final_prob_sums.sum(dim=1) == 0)     # 该样本在过滤后无有效投票
    fused_preds = final_prob_sums.argmax(dim=1)
    final_preds = torch.where(no_votes, preds0, fused_preds)  # <--- 关键回退

    if return_record:
        for i in range(B):
            for rec in records:
                if rec["sample_id"] == i:
                    rec["final_pred"] = int(final_preds[i].item())
                    rec["label"] = int(labels[i].item())
                    rec["fallback_to_origin"] = int(no_votes[i].item())  # 可选调试信息

    return final_preds, records if return_record else []







# ====== 可视化部分 ======
def show_images(imgs, titles=None):
    n = len(imgs)
    plt.figure(figsize=(4*n, 4))
    for i, img in enumerate(imgs):
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # 1. 读取一张测试图片
    img_path = "/home/lzc/transferlearning/code/DeepDG_2/data/PACS/sketch/giraffe/7361.png"   # 换成你本地的一张图片路径
    pil_img = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
    ])
    x = transform(pil_img).unsqueeze(0)  # [1,3,224,224]

    # 2. 生成变形
    x_elastic = elastic_deform(x.clone(), alpha_std=0.01, sigma=10.0)
    x_grid    = grid_distortion(x.clone(), grid_rows=3, grid_cols=3, distort_std=0.01)
    x_flip    = hflip(x.clone())

    # 3. 转换为可显示格式
    to_pil = T.ToPILImage()
    imgs = [
        to_pil(x[0].cpu()),
        to_pil(x_elastic[0].cpu()),
        to_pil(x_grid[0].cpu()),
        to_pil(x_flip[0].cpu()),
    ]
    titles = ["Original", "Elastic", "GridDistortion", "Flip"]

    # 4. 保存
    for img, title in zip(imgs, titles):
        img.save(f"{title}.jpg")