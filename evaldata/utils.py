import argparse
import numpy as np


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average, current value, and standard deviation"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1
        self.values = []  # 新增：存储所有值用于计算标准差
        self.std_dev = 0  # 新增：标准差

        self.first = 0
        self.second = 0
        self.third = 0
        self.forth = 0
        self.fifth = 0
        self.sixth = 0
        self.seventh = 0
        self.eighth = 0

    def update(self, val, n=1):
        self.val = val
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)  # 新增：记录每个值

        # 更新标准差
        if len(self.values) > 1:
            self.std_dev = np.std(self.values)
        else:
            self.std_dev = 0

        # 原有的分段计数逻辑
        if val >= 0 and val <= 0.3:
            self.first += 1
        elif val > 0.3 and val <= 0.6:
            self.second += 1
        elif val > 0.6 and val <= 0.7:
            self.third += 1
        elif val > 0.7 and val <= 0.8:
            self.forth += 1
        elif val > 0.8 and val <= 0.85:
            self.fifth += 1
        elif val > 0.85 and val <= 0.9:
            self.sixth += 1
        elif val > 0.9 and val <= 0.95:
            self.seventh += 1
        elif val > 0.95 and val <= 1:
            self.eighth += 1

    def std(self):
        """返回标准差"""
        return self.std_dev