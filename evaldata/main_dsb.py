# -*- coding: utf-8 -*-
import os
import argparse
from PIL import Image
import numpy as np
from utils import AverageMeter
from metrics import dice_coef, iou_score, get_F1, get_accuracy, get_recall, get_precision,get_specificity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 旧版参数
    parser.add_argument('--testpath', type=str, default='', help='旧版预测路径')
    parser.add_argument('--path', type=str, default='', help='旧版GT路径')
    # 新版参数
    parser.add_argument('--pred_path', type=str, default='', help='新版预测路径')
    parser.add_argument('--gt_path', type=str, default='', help='新版GT路径')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='预测结果文件后缀（默认 Test.py 生成的 .png）')
    args = parser.parse_args()

    # 兼容旧参数
    if args.pred_path == '' and args.testpath != '':
        # 如果用户用旧参数 --testpath
        args.pred_path = os.path.join(args.testpath, 'test') if os.path.isdir(os.path.join(args.testpath, 'test')) else args.testpath
    if args.gt_path == '' and args.path != '':
        args.gt_path = os.path.join(args.path, 'test', 'masks') if os.path.isdir(os.path.join(args.path, 'test', 'masks')) else args.path

    if args.pred_path == '' or args.gt_path == '':
        parser.error('必须提供 --pred_path/--gt_path 或 --testpath/--path')

    gt_files = os.listdir(args.gt_path)
    gt_files = [f for f in gt_files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.gif')]
    gt_files.sort()
    print("########################################")
    print("GT length:", len(gt_files))
    print("Pred path:", args.pred_path)
    print("GT path:", args.gt_path)
    print("########################################")

    iou = AverageMeter()
    dice = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    recall = AverageMeter()
    precision = AverageMeter()
    specificity = AverageMeter()

    for file in gt_files:
        gtfile = os.path.join(args.gt_path, file)
        basename = os.path.splitext(file)[0]
        prefile = os.path.join(args.pred_path, basename + args.suffix)

        if not os.path.exists(prefile):
            print("[Warning] 找不到预测文件:", prefile)
            continue

        gt = Image.open(gtfile).convert('L')
        pre = Image.open(prefile).convert('L')
        gt = np.asarray(gt)
        pre = np.asarray(pre)

        h, w = gt.shape[:2]
        gt = gt.reshape(h, w, 1) / 255.0
        pre = pre.reshape(h, w, 1) / 255.0

        iou.update(iou_score(pre, gt))
        dice.update(dice_coef(pre, gt))
        f1.update(get_F1(pre, gt))
        acc.update(get_accuracy(pre, gt))
        recall.update(get_recall(pre, gt))
        precision.update(get_precision(pre, gt))
        specificity.update(get_specificity(pre, gt))

    print("acc:", f"{acc.avg:.4f} ± {acc.std():.4f}")
    print("IoU:", f"{iou.avg:.4f} ± {iou.std():.4f}")
    print("Dice:", f"{dice.avg:.4f} ± {dice.std():.4f}")
    print("F1:", f"{f1.avg:.4f} ± {f1.std():.4f}")
    print("Recall:", f"{recall.avg:.4f} ± {recall.std():.4f}")
    print("Precision:", f"{precision.avg:.4f} ± {precision.std():.4f}")
    print("SP:", f"{specificity.avg:.4f} ± {specificity.std():.4f}")
