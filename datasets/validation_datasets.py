import os
import glob
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
import torch
from scipy.io import loadmat


class Dataset:
    def getlen(self, subset):
        pass

    def getitem(self, subset, index):
        pass

    def __len__(self):
        return self.getlen(self.split_name)
    
    def __getitem__(self, idx):
        return self.getitem(self.split_name, idx)


class BaseDataset(Dataset):
    def __init__(self, normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], cap_min=0.0, cap_max=100.0, desired_size=384,
                 resize_method=cv2.INTER_LINEAR, **kwargs):
        super(BaseDataset, self).__init__()
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1,1,3)
        self.std = np.array(std).reshape(1,1,3)
        self.cap_min = cap_min
        self.cap_max = cap_max
        self.split_name = 'valid'
        self.desired_size = desired_size
        self.resize_method = resize_method
        
    def normalize_img(self, img):
        img = np.asarray(img).astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        if self.normalize:
            return (img - self.mean) / self.std
        else:
            return img
    
    def get_mask(self, depth):
        mask = np.ones_like(depth).astype(np.bool)
        mask = depth > 1e-8
        mask[depth <= self.cap_min] = 0
        mask[depth >= self.cap_max] = 0
        return mask
    
    def get_size(self, image, mult=32, strategy='longest_equal'):
        source_h, source_w = image.shape[:2]
        desired_size = self.desired_size
        if source_w == source_h:
            target_h = desired_size
            target_w = desired_size
        else:
            if strategy == 'longest_equal':
                if source_h > source_w:
                    target_h = desired_size
                    target_w = (desired_size / source_h) * source_w
                else:
                    target_h = (desired_size / source_w) * source_h
                    target_w = desired_size
            elif strategy == 'smallest_equal':
                if source_h > source_w:
                    target_h = (desired_size / source_w) * source_h
                    target_w = desired_size
                else:
                    target_h = desired_size
                    target_w = (desired_size / source_h) * source_w
            else:
                raise ValueError('Unknown strategy: {:s}'.format(strategy))

        # Keeping best aspect ratio, preserving size % 32 == 0
        target_h = round(target_h / mult) * mult
        target_w = round(target_w / mult) * mult
        return target_h, target_w
    
    def validation_resize(self, image, depth, mask, desired_size, mult=32, strategy='longest_equal'):
        target_h, target_w = self.get_size(image, mult, strategy)
        image = cv2.resize(image, (target_w, target_h), interpolation=self.resize_method)
        return image, depth, mask
    
    def format_output(self, image, depth, mask):
        return {
            'image': torch.as_tensor(image).permute(2,0,1).float(),
            'depth': 1.0 / torch.as_tensor(depth).unsqueeze(0).float(), # Convert depth to disparity
            'mask': torch.as_tensor(mask).unsqueeze(0).bool(),
            'type': torch.tensor([0]),
            'depth_cap': torch.tensor([self.cap_max])
        }


TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


class SintelDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        super(SintelDataset, self).__init__(**kwargs)
        self.path = path
        self.cap_max = 72.0
        self.file_list = glob.glob(os.path.join(self.path, 'final/*/*.png'))
        self.strategy = 'longest_equal' if 'strategy' not in kwargs else kwargs['strategy']

    def getlen(self, subset):
        assert subset == 'valid'
        return len(self.file_list)

    def image_to_depth(self, name):
        new_path = os.path.abspath(name)[len(os.path.commonprefix([os.path.abspath(self.path), os.path.abspath(name)])):]
        return os.path.splitext(os.path.join(self.path, 'depth', '/'.join(new_path.split('/')[2:])))[0] + '.dpt'

    @staticmethod
    def depth_read(filename):
        """ Read depth data from file, return as numpy array. """
        f = open(filename, 'rb')
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
            TAG_FLOAT, check)
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
            width, height)
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        return depth

    def getitem(self, subset, index):
        img = self.normalize_img(Image.open(self.file_list[index]))
        depth = self.depth_read(self.image_to_depth(self.file_list[index]))
        mask = self.get_mask(depth)

        img, depth, mask = self.validation_resize(img, depth, mask.astype(np.float32), 384, strategy=self.strategy)

        return self.format_output(img, depth, mask)


class TUMDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        super(TUMDataset, self).__init__(**kwargs)
        self.path = path
        self.cap_max = 10.0
        self.list = glob.glob(os.path.join(self.path, '*.h5'))
        self.strategy = 'longest_equal' if 'strategy' not in kwargs else kwargs['strategy']

    def getlen(self, subset):
        assert subset == self.split_name
        return len(self.list)

    def getitem(self, subset, index):
        assert subset == self.split_name

        file = h5py.File(self.list[index], 'r')
        img = self.normalize_img(file.get('/gt/img_1'))
        depth = np.asarray(file.get('/gt/gt_depth')).astype(np.float32)
        mask = self.get_mask(depth)

        img, depth, mask = self.validation_resize(img, depth, mask.astype(np.float32), self.desired_size,
                                                  strategy=self.strategy)

        return self.format_output(img, depth, mask)


class NYUDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        super(NYUDataset, self).__init__(**kwargs)
        self.path = path
        self.strategy = 'longest_equal' if 'strategy' not in kwargs else kwargs['strategy']
        self.cap_max = 10.0
        f = h5py.File(self.path, 'r')
        test_idxs = loadmat('datasets/nyu-splits.mat')['testNdxs'][:, 0] - 1

        self.test_imgs = np.transpose(np.asarray(f['images'])[test_idxs], (0, 3, 2, 1))
        self.test_depth = np.transpose(np.asarray(f['rawDepths'])[test_idxs], (0, 2, 1))

    def getlen(self, subset):
        return len(self.test_imgs)

    def getitem(self, subset, index):
        img = self.normalize_img(self.test_imgs[index])[8:-8, 8:-8]
        depth = self.test_depth[index][8:-8, 8:-8]
        mask = self.get_mask(depth)

        img, depth, mask = self.validation_resize(img, depth, mask.astype(np.float32), self.desired_size,
                                                  strategy=self.strategy)
        return self.format_output(img, depth, mask)


class ETH3DDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        super(ETH3DDataset, self).__init__(**kwargs)
        self.path = path
        self.strategy = 'longest_equal' if 'strategy' not in kwargs else kwargs['strategy']
        self.cap_max = 72.0
        self.list = glob.glob(os.path.join(self.path, '*/images_out/dslr_images_undistorted/*.JPG'))

    def getlen(self, subset):
        assert subset == self.split_name
        return len(self.list)

    def getitem(self, subset, index):
        assert subset == self.split_name

        img = self.normalize_img(Image.open(self.list[index]))
        depth = np.asarray(
            Image.open(self.list[index].replace('images_out/', 'depth/').replace('.JPG', '.png'))).astype(
            np.float32) / 1000.0
        mask = self.get_mask(depth)
        img, depth, mask = self.validation_resize(img, depth, mask.astype(np.float32), self.desired_size,
                                                  strategy=self.strategy)
        return self.format_output(img, depth, mask)


class FolderDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        super(FolderDataset, self).__init__(**kwargs)
        self.path = path
        self.strategy = 'smallest_equal' if 'strategy' not in kwargs else kwargs['strategy']
        self.lst = list(Path(self.path).rglob('*.*'))
    
    def getlen(self, subset):
        return len(self.lst)
    
    def getitem(self, subset, index):
        img = self.normalize_img(np.asarray(Image.open(os.path.join(self.path, self.lst[index])).convert('RGB')))
        depth = np.zeros_like(img)
        mask = np.zeros_like(depth)
        
        img, _, _ = self.validation_resize(img, depth, mask.astype(np.float32), self.desired_size, strategy=self.strategy)
        return {'image': torch.as_tensor(img).permute(2, 0, 1).float(), 'path': str(self.lst[index])}
