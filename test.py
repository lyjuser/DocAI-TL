import argparse
import os

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from transformers import AutoProcessor, AutoModel, LayoutLMv3Processor
from models.multimodal_skip import Visual_DocAI_Net
from mmcv.runner import load_checkpoint
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
import json
import csv
import ast
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def normalize_bbox(bbox, width, height): 
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_psnr(loader, multimodal, img_text_tampering_probs_dict, save_path, save_mode,
              eval_type=None, eval_bsize=None, verbose=False):
    multimodal.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4, metric5, metric6 = 'f1_max', 'auc', 'iou', 'precision', 'recall', 'f1_mean'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_metric5 = utils.Averager()
    val_metric6 = utils.Averager()

    target_w, target_h = config['data_size'], config['data_size']
    pbar = tqdm(loader, leave=False, desc='val')
    with torch.no_grad():
        for batch in pbar:
            img_patch_path, mask_path, patch_annotation_path = batch["inp_patch_path"], batch["gt_patch_path"], batch["patch_annotation_path"]
            img_path, img_annotation_path = batch["inp_path"], batch["img_annotation_path"]
            del batch["inp_patch_path"], batch["gt_patch_path"], batch["patch_annotation_path"], batch["inp_path"], batch["img_annotation_path"]

            # ----------------------Set img into DocAI------------------------------------------
            batch_img, batch_img_WH, batch_img_bbox, batch_text, batch_label = [], [], [], [], []
            batch_text_tampering_probs, batch_tamper_bbox = [], []
            batch_patch_original_bbox, batch_patch_norm_bbox, batch_patch_text = [], [], []
            for j in img_path:
                img = Image.open(j)
                W, H = img.size
                #resize_img = img.resize((target_w, target_h))
                # img_name_suffix = j.split('\\')[-1] # Local
                img_name_suffix = j.split('/')[-1]
                batch_text_tampering_probs.append(img_text_tampering_probs_dict[img_name_suffix])
                # batch_img.append(resize_img)
                batch_img.append(img)
                batch_img_WH.append((W, H))

                # img_annotation_path = j.replace('img', 'annotation').replace('.jpg', '.json')
                img_annotation_path = j.replace('img', 'annotation').replace('.png', '.json')
                with open(img_annotation_path, encoding='utf-8') as r:
                    img_annotation = json.load(r)
                text_list, img_bbox_list, label_list = [], [], []
                for item in img_annotation["word_info"]:
                    word = item["word"]
                    if len(word) == 0:
                        continue
                    if "width" not in item:
                        x1, y1, x2, y2 = item["left"], item["top"], item["right"], item["bottom"]
                        w, h = (x2 - x1), (y2 - y1)
                    else:
                        x1, y1, w, h = item["left"], item["top"], item["width"], item["height"]
                        x2, y2 = x1 + w, y1 + h
                    if "tamper" not in item:
                        label = 0
                    else:
                        label = item["tamper"]

                    resized_x1, resized_y1, resized_x2, resized_y2 = normalize_bbox([x1, y1, x2, y2], W, H)
                    img_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])
                    text_list.append(word)
                    label_list.append(label)

                batch_img_bbox.append(img_bbox_list)
                batch_text.append(text_list)
                batch_label.append(label_list)

            if config['DocAI'] == 'Layoutlmv3':
                processor = AutoProcessor.from_pretrained("./DocAI_pretrained/Layoutlmv3", apply_ocr=False)
            elif config['DocAI'] == 'Layoutlmv2':
                processor = AutoProcessor.from_pretrained("./DocAI_pretrained/Layoutlmv2", apply_ocr=False)
            elif config['DocAI'] == 'Lilt':
                processor = LayoutLMv3Processor.from_pretrained("./DocAI_pretrained/Lilt-funsd", apply_ocr=False)

            batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label,
                                       max_length=1024, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = batch_encoding.pop('offset_mapping') 

            bbox, attention_mask = batch_encoding["bbox"].to(device), batch_encoding["attention_mask"].to(device)
            # Process img_patch
            for index, i in enumerate(patch_annotation_path):
                W, H = batch_img_WH[index]
                with open(i, encoding='utf-8') as r:
                    patch_annotation = json.load(r)
                patch_original_bbox_list, patch_norm_bbox_list, patch_text_list = [], [], []
                for item in patch_annotation["word_info"]:
                    word = item["word"]
                    if len(word) == 0:
                        continue

                    x1, y1, x2, y2 = item["Original_left"], item["Original_top"], item["Original_right"], item["Original_bottom"]
                    resized_x1, resized_y1, resized_x2, resized_y2 = normalize_bbox([x1, y1, x2, y2], W, H)
                    patch_original_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])

                    patch_x1, patch_y1, patch_x2, patch_y2 = item["left"], item["top"], item["right"], item["bottom"]
                    norm_patch_x1, norm_patch_y1, norm_patch_x2, norm_patch_y2 = patch_x1 / target_w, patch_y1 / target_h, \
                                                                                 patch_x2 / target_w, patch_y2 / target_h
                    patch_norm_bbox_list.append([norm_patch_x1, norm_patch_y1, norm_patch_x2, norm_patch_y2])
                    patch_text_list.append(word)

                batch_patch_original_bbox.append(patch_original_bbox_list)
                batch_patch_norm_bbox.append(patch_norm_bbox_list) 
                batch_patch_text.append(patch_text_list)

            max_length1 = max(len(sub_l) for sub_l in batch_patch_norm_bbox)
            batch_patch_norm_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length1 - len(sub_l)) for sub_l in batch_patch_norm_bbox]  # Padding
            batch_patch_norm_bbox_padding = torch.tensor(batch_patch_norm_bbox_padding, dtype=torch.float).to(device)

            max_length3 = max(len(sub_l) for sub_l in batch_text_tampering_probs)
            max_length2 = attention_mask.shape[1]
            max_length = max(max_length2, max_length3)
            batch_text_tampering_probs_padding = [sub_l + [0.] * (max_length - len(sub_l)) for sub_l in batch_text_tampering_probs]  # Padding
            batch_text_tampering_probs_padding = torch.tensor(batch_text_tampering_probs_padding, dtype=torch.float).unsqueeze(-1).to(device)  # (B, token_length, 1)

            max_length4 = max(len(sub_l) for sub_l in batch_patch_original_bbox)
            batch_patch_original_bbox_padding = [sub_l + [[0, 0, 0, 0]] * (max_length4 - len(sub_l)) for sub_l in batch_patch_original_bbox]  # Padding
            batch_patch_original_bbox_padding = torch.tensor(batch_patch_original_bbox_padding, dtype=torch.long).to(device)

            for k, v in batch.items():
                batch[k] = v.to(device)

            inp = batch['inp_patch']

            pred, probability_map = multimodal.infer(inp, bbox=bbox, attention_mask=attention_mask, patch_norm_bbox=batch_patch_norm_bbox_padding,
                                                     patch_original_bbox=batch_patch_original_bbox_padding, offset_mapping=offset_mapping, text_tampering_prob=batch_text_tampering_probs_padding)
            pred = torch.sigmoid(pred)
            pred_ = pred.clone()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            if save_mode == '1':
                mask = cv2.imread(mask_path[0])
                h, w, _ = mask.shape
                print("mask_path: ", mask_path)
                save_mask_path = os.path.join(save_path, mask_path[0].split('/')[-1])
                # save_mask_path = os.path.join(save_path, mask_path[0].split('\\')[-1])
                print("save_mask_path: ", save_mask_path)
                output = pred[0].permute(1, 2, 0).detach().cpu().numpy()
                output = (output * 255.).astype(np.uint8).squeeze(2)
                # output = cv2.resize(output, (w, h))
                output = Image.fromarray(output)
                output.save(save_mask_path)

                # save_previous_map_path = os.path.join(save_path, '{}_previous_map.png'.format(mask_path[0].split('/')[-1].split('.')[0]))
                # previous = previous.squeeze(dim=1).cpu().detach().numpy()  # (B, 1024, 1024)
                # plt.imsave(save_previous_map_path, previous[0], cmap='gray')

                save_probability_map_path = os.path.join(save_path, '{}_probability_map.png'.format(mask_path[0].split('/')[-1].split('.')[0]))
                probability_map = probability_map.squeeze(dim=1).cpu().detach().numpy()  # (B, 1024, 1024)
                plt.imsave(save_probability_map_path, probability_map[0], cmap='gray')

            result1, result2, result3, result4, result5, result6 = metric_fn(pred, batch['gt_patch'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])
            val_metric5.add(result5.item(), inp.shape[0])
            val_metric6.add(result6.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
                pbar.set_description('val {} {:.4f}'.format(metric5, val_metric5.item()))
                pbar.set_description('val {} {:.4f}'.format(metric6, val_metric6.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_metric5.item(), val_metric6.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--save_name')
    parser.add_argument('--csv_path')
    parser.add_argument('--save_mode', default='0')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8)

    # model = models.make(config['model']).cuda()
    visual_model = models.make(config['model'])
    # sam_checkpoint = torch.load(args.model, map_location=device)
    # print("State_dict keys:", sam_checkpoint.keys())
    # visual_model.load_state_dict(sam_checkpoint, strict=True)
    multimodal = Visual_DocAI_Net(visual_model, device=device, channel_attention=config['channel_attention']).to(device)
    sam_checkpoint = torch.load(args.model, map_location=device)
    print("State_dict keys:", sam_checkpoint.keys())
    multimodal.load_state_dict(sam_checkpoint, strict=True)

    save_path = os.path.join('results', args.save_name)
    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    csv_path = args.csv_path
    img_text_tampering_probs_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        for row in f_csv:
            img_text_tampering_probs_dict[row[0]] = ast.literal_eval(row[1])

    metric1, metric2, metric3, metric4, metric5, metric6 = eval_psnr(loader, multimodal, img_text_tampering_probs_dict,
                                                                     save_path, args.save_mode, eval_type=config.get('eval_type'),
                                                                     eval_bsize=config.get('eval_bsize'), verbose=True)
    f1_mean = (2 * metric4 * metric5) / (metric4 + metric5 + 1e-8)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    print('metric5: {:.4f}'.format(metric5))
    print('metric6: {:.4f}'.format(metric6))
    print('F1_mean: {:.4f}'.format(f1_mean))
