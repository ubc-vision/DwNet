###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os
import re

import numpy as np
import scipy.spatial

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class DensePose:
    def __init__(self, spatial_size, oob_ocluded=False, naive_warp = False):
        self.spatial_size = spatial_size
        self.oob_ocluded = oob_ocluded
        self.coordinate_grid = self.make_coordinate_grid(self.spatial_size)
        self.naive_warp = naive_warp

    def make_coordinate_grid(self, spatial_size):
        h, w = self.spatial_size
        x = np.arange(w)
        y = np.arange(h)

        x = (2.0 * (x / (w - 1.0)) - 1.0)
        y = (2.0 * (y / (h - 1.0)) - 1.0)

        xx, yy = np.meshgrid(x, y)

        meshed = np.concatenate([xx[:, :, np.newaxis], yy[:, :, np.newaxis]], 2)

        return meshed

    def nn_search(self, reference, query):
        tree = scipy.spatial.cKDTree(reference)
        _, index = tree.query(query)
        return index

    def distance(self,reference, query):
        reference = np.expand_dims(reference, axis =1)
        query = np.expand_dims(query, axis =0)
        dm = ((reference - query) ** 2).sum(-1)
        return dm


    def get_grid_warp(self, d_s, d_t):
        """
        d_s - source dence pose [h,w,3] (u, v, part_id)
        """
        warp_grid = self.coordinate_grid.copy()
        if self.oob_ocluded:
            warp_grid[d_s[:, :, 2] != 0.0] = (-1.0, -1.0)
        for part_id in range(1, 25):
            mask_s = (d_s[:, :, 2] == part_id)
            mask_t = (d_t[:, :, 2] == part_id)
            uv_s = d_s[:, :, :2][mask_s]
            uv_t = d_t[:, :, :2][mask_t]
            uv_s = uv_s.astype(float)
            uv_t = uv_t.astype(float)
            if uv_t.shape[0] == 0:
                continue
            if uv_s.shape[0] == 0:
                if self.oob_ocluded:
                    warp_grid[mask_t] = (-1, -1)
                    continue
            grid_s = self.coordinate_grid[mask_s]

            #Finding nearest neighbours
            if self.naive_warp:
                dm = self.distance(uv_s, uv_t)
                coords = grid_s[dm.argmin(axis=0)]
            else:
                index = self.nn_search(uv_s, uv_t)
                coords = grid_s[index]
            warp_grid[mask_t] = coords
        return warp_grid


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
