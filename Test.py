# -*- coding: utf-8 -*-


import os
import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from skimage import img_as_ubyte
from PIL import Image

from utils.dataloader import test_dataset

# 导入所有模型
from model.MDFANet import MDFANet



def resolve_paths(base_dir: str):
    """返回 (image_root, gt_root, roi_root 或 None)"""
    image_root = os.path.join(base_dir, 'images')
    if os.path.exists(os.path.join(base_dir, '1st_manual')):
        gt_root = os.path.join(base_dir, '1st_manual')
    else:
        gt_root = os.path.join(base_dir, 'masks')
    roi_root = os.path.join(base_dir, 'mask')
    roi_root = roi_root if os.path.isdir(roi_root) else None
    return image_root, gt_root, roi_root


def find_first_existing(path_stem: str, exts=('.png', '.gif', '.tif', '.tiff')):
    """给定路径前缀，尝试多种扩展名，返回第一个存在的路径"""
    for ext in exts:
        cand = path_stem + ext
        if os.path.exists(cand):
            return cand
    return None


def get_model_by_train_save(train_save):
    """根据 train_save 名称推断模型类型"""
    if 'noDe' in train_save:
        return MDFANet_noDe()
    elif 'noAFA' in train_save:
        return MDFANet_noAFA()
    elif 'Baseline' in train_save:
        return MDFANet_Baseline()
    elif 'noDS' in train_save:
        return MDFANet_noDS()
    elif 'noMDCF' in train_save:
        return MDFANet_noMDCF()
    else:
        return MDFANet()  # 默认


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='testing size')
    parser.add_argument('--test_path', type=str, required=True,
                        help='包含各子数据集目录的根路径，如 .../data/DRIVE')
    parser.add_argument('--train_save', type=str, required=True,
                        help='模型权重名：snapshots/<train_save>/<train_save>.pth')
    args = parser.parse_args()

    subset_names = ['test', 'CVC-300', 'CVC-ClinicDB', 'Kvasir',
                    'CVC-ColonDB', 'ETIS-LaribPolypDB']
    subdirs = [d for d in os.listdir(args.test_path)
               if os.path.isdir(os.path.join(args.test_path, d))]

    # 根据 train_save 名称自动选择模型
    model = get_model_by_train_save(args.train_save)
    print(f"Using model: {model.__class__.__name__}")

    pth_path = os.path.join('./snapshots', args.train_save, args.train_save + '.pth')
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f'找不到权重文件: {pth_path}')

    weights = torch.load(pth_path, map_location='cuda')

    # 处理状态字典
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if 'total_ops' not in k and 'total_params' not in k:
            # 处理可能的键名不匹配
            if k.startswith('module.'):
                k = k[7:]  # 移除 'module.' 前缀
            new_state_dict[k] = v

    # 加载模型权重
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"加载权重失败: {e}")
        print("尝试使用 strict=False 加载...")
        model.load_state_dict(new_state_dict, strict=False)

    model.cuda().eval()

    with torch.no_grad():
        for name in subset_names:
            if name not in subdirs:
                continue

            data_path = os.path.join(args.test_path, name)
            print("#" * 30)
            print("Now test dir is:", data_path)
            print("#" * 30)
            time.sleep(0.3)

            save_path = os.path.join(
                './Result', args.train_save,
                f"{os.path.basename(os.path.dirname(data_path))}", name
            )
            os.makedirs(save_path, exist_ok=True)

            image_root, gt_root, roi_root = resolve_paths(data_path)
            loader = test_dataset(image_root, gt_root, args.testsize)

            for _ in range(loader.size):
                image, gt, base_name = loader.load_data()
                gt = np.asarray(gt, np.float32)
                h, w = gt.shape[:2]

                image = image.cuda()
                outputs = model(image)

                # 处理不同模型的输出
                if isinstance(outputs, (list, tuple)):
                    if len(outputs) >= 6:
                        res = outputs[0]  # 有深度监督的模型，取第一个输出
                    else:
                        res = outputs[0]  # 其他情况取第一个
                else:
                    res = outputs  # 单个输出

                # 上采样并归一化
                res = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                res = torch.sigmoid(res).data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                # ROI 裁剪
                if roi_root is not None:
                    stem = os.path.splitext(base_name)[0]
                    roi_cand = find_first_existing(os.path.join(roi_root, stem))
                    if roi_cand is not None:
                        roi_img = Image.open(roi_cand).convert('L')
                        roi = np.asarray(roi_img, np.float32) / 255.0
                        if roi.shape[:2] != (h, w):
                            roi_img = roi_img.resize((w, h), Image.NEAREST)
                            roi = np.asarray(roi_img, np.float32) / 255.0
                        res = res * roi

                # ==== 保存命名：保证是 *_test_mask.gif ====
                root, _ = os.path.splitext(base_name)
                if root.endswith('_test'):
                    out_name = root + '_mask.gif'
                elif root.endswith('_lesion'):
                    out_name = root + '.bmp'
                else:
                    out_name = root + '.png'

                out_path = os.path.join(save_path, out_name)
                print("Saving result to:", out_path)
                imageio.imwrite(out_path, img_as_ubyte(res))


if __name__ == '__main__':
    main()