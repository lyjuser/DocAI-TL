import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register


# ----------------------------------Patch for SAM and Image for DocAI--------------------------------------
# @register('image-folder')
# class ImageFolder(Dataset):
#     def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
#                  repeat=1, cache='none', mask=False):
#         self.repeat = repeat
#         self.cache = cache
#         self.path = path
#         self.Train = False
#         self.split_key = split_key
#
#         self.size = size
#         self.mask = mask
#         if self.mask:
#             self.img_transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#         else:
#             self.img_transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#
#         if split_file is None:
#             filenames = sorted(os.listdir(path))
#         else:
#             with open(split_file, 'r') as f:
#                 filenames = json.load(f)[split_key]
#         if first_k is not None:
#             filenames = filenames[:first_k]
#
#         self.files = []
#
#         for filename in filenames:
#             file = os.path.join(path, filename)
#             self.append_file(file)
#
#     def append_file(self, file):
#         if self.cache == 'none':
#             self.files.append(file)
#         elif self.cache == 'in_memory':
#             self.files.append(self.img_process(file))
#
#     def __len__(self):
#         return len(self.files) * self.repeat
#
#     def __getitem__(self, idx):
#         x = self.files[idx % len(self.files)]
#         self.name = x
#
#         if self.cache == 'none':
#             return (self.img_process(x), self.name)
#         elif self.cache == 'in_memory':
#             return x
#
#     def img_process(self, file):
#         if self.mask:
#             return Image.open(file).convert('L')
#         else:
#             return Image.open(file).convert('RGB')
#
# @register('paired-image-folders')
# class PairedImageFolders(Dataset):
#
#     def __init__(self, root_path_1, root_path_2, **kwargs):
#         self.dataset_1 = ImageFolder(root_path_1, **kwargs)
#         self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
#
#     def __len__(self):
#         return len(self.dataset_1)
#
#     def __getitem__(self, idx):
#         img, img_path = self.dataset_1[idx]
#         mask, mask_path = self.dataset_2[idx]
#         return img, img_path, mask, mask_path
#         # return self.dataset_1[idx], self.dataset_2[idx]


# add annotation_path
@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename) # img_path_list
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        self.name = x

        if self.cache == 'none':
            return (self.img_process(x), self.name)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

class ImageLoader(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename) # img_list
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, img_patch_path):
        # img_patch_name = img_patch_path.split('\\')[-1] # Local
        img_patch_name = img_patch_path.split('/')[-1] # Linux
        if len(img_patch_name.split('.')[0].split('_')) == 2: # For T-SROIE
            img_name = '{}.jpg'.format(img_patch_name.split('.')[0].split('_')[0])
        elif len(img_patch_name.split('.')[0].split('_')) == 3:  # For IDC
            img_name = '{}_{}.jpg'.format(img_patch_name.split('.')[0].split('_')[0], img_patch_name.split('.')[0].split('_')[1])
        else:
            img_name = '{}_{}_{}_{}.jpg'.format(img_patch_name.split('.')[0].split('_')[0], img_patch_name.split('.')[0].split('_')[1],
                                            img_patch_name.split('.')[0].split('_')[2], img_patch_name.split('.')[0].split('_')[3])
            # For Funsd
            # img_name = '{}_{}_{}_{}.png'.format(img_patch_name.split('.')[0].split('_')[0], img_patch_name.split('.')[0].split('_')[1],
            #                                 img_patch_name.split('.')[0].split('_')[2], img_patch_name.split('.')[0].split('_')[3])
        img_path = os.path.join(self.path, img_name)
        idx = self.files.index(img_path)
        x = self.files[idx % len(self.files)]
        self.name = x

        if self.cache == 'none':
            # return (self.img_process(x), self.name, idx)
            return (self.name, idx)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, root_path_5, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
        self.patch_annotation_name_list = []
        filenames = sorted(os.listdir(root_path_3))
        for filename in filenames:
            file = os.path.join(root_path_3, filename) # annotation_path_list
            self.patch_annotation_name_list.append(file)

        self.dataset_3 = ImageLoader(root_path_4, **kwargs)
        self.img_annotation_name_list = []
        filenames = sorted(os.listdir(root_path_5))
        for filename in filenames:
            file = os.path.join(root_path_5, filename) # annotation_img_list
            self.img_annotation_name_list.append(file)


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        img_patch, img_patch_path = self.dataset_1[idx]
        mask_patch, mask_patch_path = self.dataset_2[idx]
        patch_annotation_path = self.patch_annotation_name_list[idx]

        # img, img_path, index = self.dataset_3[img_patch_path]
        img_path, index = self.dataset_3[img_patch_path]
        img_annotation_path = self.img_annotation_name_list[index]


        return img_patch, img_patch_path, mask_patch, mask_patch_path, patch_annotation_path, \
               img_path, img_annotation_path
