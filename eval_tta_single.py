# coding=utf-8
import os
import torch
from train import get_args
import numpy as np
import math
from alg import modelopera,alg
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from datautil.imgdata.util import rgb_loader, l_loader
from tps import build_tta_views, tta_predict_conf, tta_predict_softmax
import matplotlib.pyplot as plt
import pandas as pd



class PairedCartoonDataset(Dataset):
    """返回 (cartoon_img, edge_img, label)"""
    def __init__(self, cartoon_paths, transform=None):
        self.cartoon = ImageFolder(cartoon_paths).imgs


        cimgs = [item[0] for item in self.cartoon]
        clabels = [item[1] for item in self.cartoon]
        self.clabels = np.array(clabels)
        self.x = cimgs

        
        self.transform     = transform
        self.loader = rgb_loader

    def __len__(self):

        return len(self.clabels)

    def __getitem__(self, idx):
        img_path   = self.x[idx]
        
        
        img       = self.loader(img_path)   
        label     = self.clabels[idx]
        
        if self.transform:
            img      = self.transform(img)
        return img, label




def plot_confidence_histogram(all_records, save_path: str, target_domain: str, is_edge: bool, bins=20):
    """
    绘制正确预测和错误预测置信度直方图（按百分比显示），并保存到文件。

    参数:
        all_records: list[dict] 或 pd.DataFrame，包含至少字段 ['conf', 'correct', 'final_pred']
        save_path: str，保存图像路径
        target_domain: str，目标域名称，用于标题
        is_edge: bool，是否为 edge 样本，用于标题
        bins: int，直方图分箱数量
    """
    # 如果传入 list，自动转换成 DataFrame
    if isinstance(all_records, list):
        all_records = pd.DataFrame(all_records)

    # 只保留有效预测
    df = all_records[all_records['final_pred'] != -1]

    # 正确预测置信度
    all_correct = df[df['correct'] == 1]['conf'].values
    # 错误预测置信度
    all_wrong = df[df['correct'] == 0]['conf'].values

    plt.figure(figsize=(10,4))

    # ===== 正确预测 =====
    counts, bin_edges = np.histogram(all_correct, bins=bins, range=(0,1))
    percents = counts / counts.sum() * 100
    plt.subplot(1,2,1)
    plt.bar(bin_edges[:-1], percents, width=np.diff(bin_edges), align="edge", color="g", alpha=0.7)
    plt.title(f"{target_domain} | {'edge' if is_edge else 'normal'} | Correct prediction")
    plt.xlabel("Confidence")
    plt.ylabel("Percentage (%)")

    # ===== 错误预测 =====
    counts, bin_edges = np.histogram(all_wrong, bins=bins, range=(0,1))
    percents = counts / counts.sum() * 100
    plt.subplot(1,2,2)
    plt.bar(bin_edges[:-1], percents, width=np.diff(bin_edges), align="edge", color="r", alpha=0.7)
    plt.title(f"{target_domain} | {'edge' if is_edge else 'normal'} | Incorrect prediction")
    plt.xlabel("Confidence")
    plt.ylabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confidence histogram to {save_path}")


def main():
    # 复用train.py的参数解析
    args = get_args()

    # origin_paths = "/home/lzc/transferlearning/code/DeepDG/data/PACS/sketch"
    # # origin_paths = "/home/lzc/transferlearning/code/DeepDG_2/data/pacs_c/photo_c"
    # edge_paths    = "/home/lzc/transferlearning/code/DeepDG_2/data/pacs_c/sketch_c"
    origin_paths = "/home/lzc/transferlearning/code/DeepDG_2/data/OfficeHome/Art"


    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    test_ds = PairedCartoonDataset(origin_paths, transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    
    # 加载模型
    output1 = "/home/lzc/transferlearning/code/DeepDG/officehome/resnet50/a/erm"


    model_path1 = os.path.join(output1, 'best_model.pkl')
    if not os.path.exists(model_path1):
        raise FileNotFoundError(f"Model file {model_path1} not found")
    
    # 加载保存的模型字典
    checkpoint1 = torch.load(model_path1)
    # 重建模型对象
    algorithm_class1 = alg.get_algorithm_class(args.algorithm)
    algorithm1 = algorithm_class1(args).cuda()
    # print(algorithm.discriminator.layers)
    # 加载模型参数
    algorithm1.load_state_dict(checkpoint1['model_dict'],strict=False)
    algorithm1.eval()



    # ---------- TTA 超参（可调） ----------
    tta_cfg = dict(
        do_elastic=True,
        do_grid=True,
        elastic_params=dict(alpha_std=0.005, sigma=10.0),  # 弹性幅度/平滑
        grid_params=dict(grid_rows=3, grid_cols=3, distort_std=0.005),  # 网格密度/幅度
        include_flip=True,
    )
   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取测试环境名称

    correct_origin                     = 0
    correct_origin_tta_without_cf      = 0
    correct_origin_sp_tta              = 0
    total                              = 0 




    with torch.no_grad():
        for batch_idx, (origin, clabels) in enumerate(test_loader):
            origin = origin.to(device)
            label   = clabels.to(device)

            # 生成 TTA 视图
            origin_views = build_tta_views(origin, **tta_cfg)


            #  单视图（原图） logits，用于“单模型不带 TTA”的对比统计
            logits1_single = algorithm1.predict(origin)  # 原图
            

            # # TTA 概率平均（softmax 后平均）
            p1 = tta_predict_softmax(algorithm1, origin, origin_views)  # [B,C]
            preb1_tta, _ = tta_predict_conf(algorithm1, origin, origin_views, label, conf_thres=0.7, return_record=True, batch_id=batch_idx)
            
            pred1_single = logits1_single.argmax(dim=1)
            correct_origin += (pred1_single == label).sum().item()

            correct_origin_tta_without_cf += (p1.argmax(dim=1) == label).sum().item()

            correct_origin_sp_tta   += (preb1_tta == label).sum().item()


            total += label.size(0)

    acc_origin = correct_origin / total

    acc_origin_tta_without_cf = correct_origin_tta_without_cf / total

    acc_origin_sp_tta = correct_origin_sp_tta / total


    print(f'Origin-only accuracy : {acc_origin*100:5.2f}%')

    print(f'Origin-TTA-without CF  accuracy : {acc_origin_tta_without_cf*100:5.2f}%')
    print(f'Origin-SP-TTA accuracy : {acc_origin_sp_tta*100:5.2f}%')


    # 保存 origin
    # df1 = pd.DataFrame(all_records_origin)
    # df1.to_csv("tta_results_origin_art_std001_50_office_mmd.csv", index=False)


    

if __name__ == '__main__':
    main()


