from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from .transforms import Compose,RandomHorizontallyFlip
# import transforms as T
# import ..transforms as F

class Preprocessor_tran(Dataset):
    def __init__(self, dataset, root=None, main_transform=None, img_transform=None, gt_transform=None):
        super(Preprocessor_tran, self).__init__()
        self.dataset = dataset
        self.root = root
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        # 随机以给定的概率 p 应用一个转换列表。在这里，使用了 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) 进行颜色抖动，概率为 0.8。
        # transforms.RandomGrayscale：以给定的概率 p 随机将图像转换为灰度图像。在这里，概率为 0.2。
        self.base_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2)
        ])
        # RandomHorizontallyFlip() 数据增强函数之一，用于随机水平翻转图像。这个函数会以一定的概率随机地水平翻转输入的图像。
        self.random_flip = Compose([RandomHorizontallyFlip()])
        # self.random_flip = T.Compose([T.RandomHorizontallyFlip()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root+'/img', fname)
        
        img = Image.open(fpath).convert('RGB')
        
        den = pd.read_csv(os.path.join(self.root+'/den', os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        if self.main_transform is not None:
            img, den = self.main_transform(img,den)
        img2 = self.base_transform(img)
        img2, den2 = self.random_flip(img2, den)
 
        if self.img_transform is not None:
            img = self.img_transform(img)
            img2 = self.img_transform(img2)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
            den2 = self.gt_transform(den2)

        return img, den, img2, den2