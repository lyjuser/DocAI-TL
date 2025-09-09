import os
import json
from PIL import Image
import argparse
import pickle
import imageio
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register
from transformers import AutoProcessor, AutoModel, AutoTokenizer, LayoutLMv3ForTokenClassification, LiltForTokenClassification, LayoutLMv3Processor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.filenames = filenames

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

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        y = self.filenames[idx % len(self.filenames)]
        self.img_path = x
        self.img_name = y

        if self.cache == 'none':
            # return (self.img_process(x), self.name, idx)
            return (self.img_path, self.img_name)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageLoader(root_path_1, **kwargs)
        self.img_annotation_name_list = []
        filenames = sorted(os.listdir(root_path_2))
        for filename in filenames:
            file = os.path.join(root_path_2, filename) # annotation_img_list
            self.img_annotation_name_list.append(file)


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):

        img_path, img_name = self.dataset_1[idx]
        img_annotation_path = self.img_annotation_name_list[idx]

        return img_path, img_name, img_annotation_path

@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, img_name, img_annotation_path = self.dataset[idx]
        return {
            'inp_path': img_path,
            'inp_name': img_name,
            'img_annotation_path': img_annotation_path
        }

import datasets
import utils
import yaml
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader

def Corrdinate_correction(x1, y1, x2, y2):
    # 防止超出[0,1023]的范围
    if y2 >= 1023:
        y2 = 1023
    if x2 >= 1023:
        x2 = 1023
    if y1 <= 0:
        y1 = 0
    if x1 <= 0:
        x1 = 0
    return x1, y1, x2, y2

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    #     log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=False, num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    return train_loader

def DocAI_for_text_tampering_probability_prediction(train_loader, DocAI):
    target_w, target_h = config['data_size'], config['data_size']
    save_csv_path = config['save_csv_path']
    for batch in tqdm(train_loader):
        img_path, img_name, img_annotation_path = batch["inp_path"], batch["inp_name"], batch["img_annotation_path"]
        print("img_path", img_path)

        batch_img, batch_img_ratio, batch_img_bbox, batch_text, batch_label = [], [], [], [], []
        for j in img_path:
            img = Image.open(j)
            W, H = img.size
            resize_img = img.resize((target_w, target_h))
            batch_img.append(resize_img)
            batch_img_ratio.append((target_w / W, target_h / H))

        for index, i in enumerate(img_annotation_path):
            ratio = batch_img_ratio[index]
            with open(i, encoding='utf-8') as r:
                img_annotation = json.load(r)
            text_list, img_bbox_list, label_list = [], [], []
            for item in img_annotation["word_info"]:
                if "width" not in item:
                    x1, y1, x2, y2 = item["left"], item["top"], item["right"], item["bottom"]
                    w, h = (x2 - x1), (y2 - y1)
                else:
                    x1, y1, w, h = item["left"], item["top"], item["width"], item["height"]
                    x2, y2 = x1 + w, y1 + h
                word = item["word"]
                if "tamper" not in item:
                    label = 0
                else:
                    label = item["tamper"]

                resized_x1, resized_y1, resized_x2, resized_y2 = int(x1 * ratio[0]), int(y1 * ratio[1]), int(x2 * ratio[0]), int(y2 * ratio[1])
                resized_x1, resized_y1, resized_x2, resized_y2 = Corrdinate_correction(resized_x1, resized_y1, resized_x2, resized_y2)
                img_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])
                text_list.append(word)
                label_list.append(label)

            batch_img_bbox.append(img_bbox_list)
            batch_text.append(text_list)
            batch_label.append(label_list)

        if config['DocAI'] == 'Layoutlmv3':
            processor = AutoProcessor.from_pretrained("E:/Visual-DocAI-Net/DocAI_pretrained/Layoutlmv3", apply_ocr=False)
            batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label,
                                       padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = batch_encoding.pop('offset_mapping')  # 返回字符偏移量映射表，长度等于文本段数目的列表，列表中每一项是一个长度等于段中字符数目的整数列表，用于表示每一个字符在原始文本中的偏移量，例如"Teacher" -> "GTE", "ACH","ER"

            input_ids, bbox, attention_mask, pixel_values = batch_encoding["input_ids"].to(device), batch_encoding["bbox"].to(device), \
                                                            batch_encoding["attention_mask"].to(device), batch_encoding["pixel_values"].to(device)

            output = DocAI(input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=pixel_values)
        elif config['DocAI'] == 'Lilt':
            processor = LayoutLMv3Processor.from_pretrained("E:/Visual-DocAI-Net/DocAI_pretrained/LilT", apply_ocr=False)
            batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label,
                                       padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = batch_encoding.pop('offset_mapping')  # 返回字符偏移量映射表，长度等于文本段数目的列表，列表中每一项是一个长度等于段中字符数目的整数列表，用于表示每一个字符在原始文本中的偏移量，例如"Teacher" -> "GTE", "ACH","ER"

            input_ids, bbox, attention_mask, pixel_values = batch_encoding["input_ids"].to(device), batch_encoding["bbox"].to(device), \
                                                            batch_encoding["attention_mask"].to(device), batch_encoding["pixel_values"].to(device)

            output = DocAI(input_ids, bbox=bbox, attention_mask=attention_mask)

        text_tampering_probs = F.softmax(output.logits, dim=-1)[:, :, 5].tolist() # (B, token_length)
        save_result = list(zip(img_name, text_tampering_probs))
        with open(save_csv_path, 'a+', encoding='utf-8', newline='') as f:
            header = ['img_name', 'text_tampering_probs']
            f_csv = csv.writer(f)
            if f.tell() == 0:
                f_csv.writerow(header)
            f_csv.writerows(save_result)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loaders()
    if config['DocAI'] == 'Layoutlmv3':
        DocAI = LayoutLMv3ForTokenClassification.from_pretrained(config['DocAI_checkpoint']).to(device)
        DocAI_for_text_tampering_probability_prediction(train_loader, DocAI)
    elif config['DocAI'] == 'Lilt':
        DocAI = LiltForTokenClassification.from_pretrained(config['DocAI_checkpoint']).to(device)
        DocAI_for_text_tampering_probability_prediction(train_loader, DocAI)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)