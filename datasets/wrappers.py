
import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F
#-------------------- Patch for SAM and DocAI --------------------------------------------
# def to_mask(mask):
#     return transforms.ToTensor()(
#         transforms.Grayscale(num_output_channels=1)(
#             transforms.ToPILImage()(mask)))
#
#
# def resize_fn(img, size):
#     return transforms.ToTensor()(
#         transforms.Resize(size)(
#             transforms.ToPILImage()(img)))
#
#
# @register('val')
# class ValDataset(Dataset):
#     def __init__(self, dataset, inp_size=None, augment=False):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.augment = augment
#
#         self.img_transform = transforms.Compose([
#                 transforms.Resize((inp_size, inp_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.mask_transform = transforms.Compose([
#                 transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         img, img_path, mask, mask_path = self.dataset[idx]
#
#         return {
#             'inp': self.img_transform(img),
#             'inp_path': img_path,
#             'gt': self.mask_transform(mask),
#             'gt_path': mask_path
#         }
#
#
# @register('train')
# class TrainDataset(Dataset):
#     def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
#                  augment=False, gt_resize=None):
#         self.dataset = dataset
#         self.size_min = size_min
#         if size_max is None:
#             size_max = size_min
#         self.size_max = size_max
#         self.augment = augment
#         self.gt_resize = gt_resize
#
#         self.inp_size = inp_size
#         self.img_transform = transforms.Compose([
#                 transforms.Resize((self.inp_size, self.inp_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.inverse_transform = transforms.Compose([
#                 transforms.Normalize(mean=[0., 0., 0.],
#                                      std=[1/0.229, 1/0.224, 1/0.225]),
#                 transforms.Normalize(mean=[-0.485, -0.456, -0.406],
#                                      std=[1, 1, 1])
#             ])
#         self.mask_transform = transforms.Compose([
#                 transforms.Resize((self.inp_size, self.inp_size)),
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         # img, mask = self.dataset[idx]
#         img, img_path, mask, mask_path = self.dataset[idx]
#
#         # random filp
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#
#         img = transforms.Resize((self.inp_size, self.inp_size))(img)
#         mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
#
#         # return {
#         #     'inp': self.img_transform(img),
#         #     'gt': self.mask_transform(mask)
#         # }
#         return {
#             'inp': self.img_transform(img),
#             'inp_path': img_path,
#             'gt': self.mask_transform(mask),
#             'gt_path': mask_path
#         }


# add annotation_path
# def to_mask(mask):
#     return transforms.ToTensor()(
#         transforms.Grayscale(num_output_channels=1)(
#             transforms.ToPILImage()(mask)))
#
#
# def resize_fn(img, size):
#     return transforms.ToTensor()(
#         transforms.Resize(size)(
#             transforms.ToPILImage()(img)))
#
#
# @register('val')
# class ValDataset(Dataset):
#     def __init__(self, dataset, inp_size=None, augment=False):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.augment = augment
#
#         self.img_transform = transforms.Compose([
#                 transforms.Resize((inp_size, inp_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.mask_transform = transforms.Compose([
#                 transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     # def __getitem__(self, idx):
#     #     img, mask = self.dataset[idx]
#     #
#     #     return {
#     #         'inp': self.img_transform(img),
#     #         'gt': self.mask_transform(mask)
#     #     }
#
#     def __getitem__(self, idx):
#         img, img_path, mask, mask_path, annotation_path = self.dataset[idx]
#
#         return {
#             'inp': self.img_transform(img),
#             'inp_path': img_path,
#             'gt': self.mask_transform(mask),
#             'gt_path': mask_path,
#             'annotation_path': annotation_path
#         }
#
#
#
# @register('train')
# class TrainDataset(Dataset):
#     def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
#                  augment=False, gt_resize=None):
#         self.dataset = dataset
#         self.size_min = size_min
#         if size_max is None:
#             size_max = size_min
#         self.size_max = size_max
#         self.augment = augment
#         self.gt_resize = gt_resize
#
#         self.inp_size = inp_size
#         self.img_transform = transforms.Compose([
#                 transforms.Resize((self.inp_size, self.inp_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.inverse_transform = transforms.Compose([
#                 transforms.Normalize(mean=[0., 0., 0.],
#                                      std=[1/0.229, 1/0.224, 1/0.225]),
#                 transforms.Normalize(mean=[-0.485, -0.456, -0.406],
#                                      std=[1, 1, 1])
#             ])
#         self.mask_transform = transforms.Compose([
#                 transforms.Resize((self.inp_size, self.inp_size)),
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         # img, mask = self.dataset[idx]
#         img, img_path, mask, mask_path, annotation_path = self.dataset[idx]
#
#         # random filp
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#
#         img = transforms.Resize((self.inp_size, self.inp_size))(img)
#         mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
#
#         # return {
#         #     'inp': self.img_transform(img),
#         #     'gt': self.mask_transform(mask)
#         # }
#         return {
#             'inp': self.img_transform(img),
#             'inp_path': img_path,
#             'gt': self.mask_transform(mask),
#             'gt_path': mask_path,
#             'annotation_path': annotation_path
#         }


# -----------------------Patch for SAM and Image for DocAI----------------------------------------
# def to_mask(mask):
#     return transforms.ToTensor()(
#         transforms.Grayscale(num_output_channels=1)(
#             transforms.ToPILImage()(mask)))
#
#
# def resize_fn(img, size):
#     return transforms.ToTensor()(
#         transforms.Resize(size)(
#             transforms.ToPILImage()(img)))
#
#
# @register('val')
# class ValDataset(Dataset):
#     def __init__(self, dataset, inp_size=None, augment=False):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.augment = augment
#
#         self.img_transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.mask_transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         img, img_path, mask, mask_path = self.dataset[idx]
#
#         return {
#             'inp': self.img_transform(img),
#             'inp_path': img_path,
#             'gt': self.mask_transform(mask),
#             'gt_path': mask_path
#         }
#
#
# @register('train')
# class TrainDataset(Dataset):
#     def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
#                  augment=False, gt_resize=None):
#         self.dataset = dataset
#         self.size_min = size_min
#         if size_max is None:
#             size_max = size_min
#         self.size_max = size_max
#         self.augment = augment
#         self.gt_resize = gt_resize
#
#         self.inp_size = inp_size
#         self.img_transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.inverse_transform = transforms.Compose([
#                 transforms.Normalize(mean=[0., 0., 0.],
#                                      std=[1/0.229, 1/0.224, 1/0.225]),
#                 transforms.Normalize(mean=[-0.485, -0.456, -0.406],
#                                      std=[1, 1, 1])
#             ])
#         self.mask_transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         # img, mask = self.dataset[idx]
#         img, img_path, mask, mask_path = self.dataset[idx]
#
#         # random filp
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#
#         # return {
#         #     'inp': self.img_transform(img),
#         #     'gt': self.mask_transform(mask)
#         # }
#         return {
#             'inp': self.img_transform(img),
#             'inp_path': img_path,
#             'gt': self.mask_transform(mask),
#             'gt_path': mask_path
#         }

# add annotation_path
def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img_patch, img_patch_path, mask_patch, mask_patch_path, patch_annotation_path, img, img_path, img_annotation_path = self.dataset[idx]
        img_patch, img_patch_path, mask_patch, mask_patch_path, patch_annotation_path, img_path, img_annotation_path = self.dataset[idx]

        return {
            'inp_patch': self.img_transform(img_patch),
            'inp_patch_path': img_patch_path,
            'gt_patch': self.mask_transform(mask_patch),
            'gt_patch_path': mask_patch_path,
            'patch_annotation_path': patch_annotation_path,

            'inp_path': img_path,
            'img_annotation_path': img_annotation_path
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img_patch, img_patch_path, mask_patch, mask_patch_path, patch_annotation_path, img, img_path, img_annotation_path = self.dataset[idx]
        img_patch, img_patch_path, mask_patch, mask_patch_path, patch_annotation_path, img_path, img_annotation_path = self.dataset[idx]
        # random filp
        # if random.random() < 0.5:
        #     img_patch = img_patch.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask_patch = mask_patch.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            'inp_patch': self.img_transform(img_patch),
            'inp_patch_path': img_patch_path,
            'gt_patch': self.mask_transform(mask_patch),
            'gt_patch_path': mask_patch_path,
            'patch_annotation_path': patch_annotation_path,
            'inp_path': img_path,
            'img_annotation_path': img_annotation_path
        }