import os

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch

from .validation_datasets import BaseDataset


def read_diw_anno(path):
    lines = [f[:-1] for f in open(path, 'r').readlines()]
    paths, points = lines[::2], lines[1::2]
    return pd.DataFrame([tuple([paths[i]] + points[i].split(',')) for i in range(len(paths))], 
                        columns=['path', 'yA', 'xA', 'yB', 'xB', 'rel', 'w', 'h'])


class DIW(BaseDataset):
    def __init__(self, path, **kwargs):
        super(DIW, self).__init__(**kwargs)
        self.path = path
        self.test_list = read_diw_anno(os.path.join(self.path, 'DIW_test.csv'))

    def getlen(self, subset):
        return len(self.test_list)
    
    def getitem(self, subset, index):
        path, yA, xA, yB, xB, rel, w, h = self.test_list.loc[index]
        yA, xA, yB, xB = float(yA)-1, float(xA)-1, float(yB)-1, float(xB)-1
        
        img = Image.open(os.path.join(self.path, path[2:])).convert('RGB')
        img = np.asarray(img)
        img_h, img_w = img.shape[:2]
        
        xA = np.floor(np.clip(xA, 0, img_w-1))
        yA = np.floor(np.clip(yA, 0, img_h-1))
        xB = np.floor(np.clip(xB, 0, img_w-1))
        yB = np.floor(np.clip(yB, 0, img_h-1))
        
        img = self.normalize_img(img)
        depth = torch.zeros(1, img.shape[0], img.shape[1], dtype=torch.float32)
        mask = torch.ones(1, img.shape[0], img.shape[1], dtype=torch.bool)
        target_h, target_w = self.get_size(img, mult=32, strategy='smallest_equal')
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        item = {
            'image': torch.as_tensor(img).permute(2, 0, 1).float(),
            'depth': depth,
            'mask': mask,
            'ordinal': torch.tensor([yA, xA, yB, xB, 1 if rel == '>' else -1]).long().view(1, -1),
            'type': torch.tensor([2])
        }
        
        return item
