import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
from utils import AverageMeter
from metrics import dice_coef, iou_score, get_F1, get_accuracy, get_recall, get_precision, get_specificity

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--testpath', type=str, default="")
    parse.add_argument('--path', type=str, default="")

    args = parse.parse_args()

    for _dataname in ['test']:
        gtpath = os.path.join(args.path, _dataname, "1st_manual")
        prepath = os.path.join(args.testpath, _dataname)
        files = os.listdir(gtpath)
        lens = len(files)
        print("##" * 20)
        print(_dataname, "length: ", lens)
        print("##" * 20)
        iou = AverageMeter()
        dice = AverageMeter()
        f1 = AverageMeter()
        acc = AverageMeter()
        recall = AverageMeter()
        precision = AverageMeter()
        sp = AverageMeter()

        for file in files:
            gtfile = os.path.join(gtpath, file)
            pred_file = file.replace('.gif', '.png')
            prefile = os.path.join(prepath, pred_file)

            if not os.path.exists(prefile):
                print(f"Skip: {pred_file} not found")
                continue

            gt = Image.open(gtfile).convert('L')
            pre = Image.open(prefile).convert('L')

            # 修复：调整预测图像尺寸匹配GT
            if pre.size != gt.size:
                pre = pre.resize(gt.size, Image.NEAREST)

            gt = np.asarray(gt)
            pre = np.asarray(pre)
            h, w = gt.shape
            gt = gt.reshape(h, w, 1) / 255.0
            pre = pre.reshape(h, w, 1) / 255.0

            iou.update(iou_score(pre, gt))
            dice.update(dice_coef(pre, gt))
            f1.update(get_F1(pre, gt))
            acc.update(get_accuracy(pre, gt))
            recall.update(get_recall(pre, gt))
            precision.update(get_precision(pre, gt))
            sp.update(get_specificity(pre, gt))

        print("acc:", f"{acc.avg:.4f} ± {acc.std():.4f}")
        print("Iou:", f"{iou.avg:.4f} ± {iou.std():.4f}")
        print("Dice:", f"{dice.avg:.4f} ± {dice.std():.4f}")
        print("f1:", f"{f1.avg:.4f} ± {f1.std():.4f}")
        print("recall:", f"{recall.avg:.4f} ± {recall.std():.4f}")
        print("precision:", f"{precision.avg:.4f} ± {precision.std():.4f}")
        print("SP:", f"{sp.avg:.4f} ± {sp.std():.4f}")