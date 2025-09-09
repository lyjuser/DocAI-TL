# # DocAI输入为Image SAM输入为Patch
# import argparse
# import os
#
# import yaml
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
#
# import datasets
# import models
# import utils
# from statistics import mean
# import torch
# import copy
# import json
# import csv
# import ast
# from PIL import Image
# import torch.distributed as dist
# from models.multimodal_skip import Visual_DocAI_Net
# from transformers import AutoProcessor, AutoModel, LayoutLMv3Processor, LayoutLMv2Processor
#
# torch.distributed.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
#
# def Corrdinate_correction(x1, y1, x2, y2):
#     # 防止超出[0,1023]的范围
#     if y2 >= 1023:
#         y2 = 1023
#     if x2 >= 1023:
#         x2 = 1023
#     if y1 <= 0:
#         y1 = 0
#     if x1 <= 0:
#         x1 = 0
#     return x1, y1, x2, y2
#
# def normalize_bbox(bbox, width, height): # 归一化边框
#     return [
#         int(1000 * (bbox[0] / width)),
#         int(1000 * (bbox[1] / height)),
#         int(1000 * (bbox[2] / width)),
#         int(1000 * (bbox[3] / height)),
#     ]
#
# class Averager():
#
#     def __init__(self):
#         self.n = 0.0
#         self.v = 0.0
#
#     def add(self, v, n=1.0):
#         self.v = (self.v * self.n + v * n) / (self.n + n)
#         self.n += n
#
#     def item(self):
#         return self.v
#
#
# def make_data_loader(spec, tag=''):
#     if spec is None:
#         return None
#
#     dataset = datasets.make(spec['dataset'])
#     dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
#     if local_rank == 0:
#         log('{} dataset: size={}'.format(tag, len(dataset)))
#     # for k, v in dataset[0].items():
#     #     log('  {}: shape={}'.format(k, tuple(v.shape)))
#
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#     loader = DataLoader(dataset, batch_size=spec['batch_size'],
#                         shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
#     return loader
#
#
# def make_data_loaders():
#     train_loader = make_data_loader(config.get('train_dataset'), tag='train')
#     val_loader = make_data_loader(config.get('val_dataset'), tag='val')
#     return train_loader, val_loader
#
#
# def eval_psnr(loader, multimodal, processor, img_text_tampering_probs_dict, eval_type=None): # 读取CSV文件中存储的文本篡改概率
#     multimodal.eval()
#     target_w, target_h = config['data_size'], config['data_size']
#
#     if eval_type == 'f1':
#         metric_fn = utils.calc_f1
#         metric1, metric2, metric3, metric4, metric5, metric6 = 'f1_max', 'auc', 'iou', 'p', 'r', 'f1_mean'
#
#     val_metric1 = Averager()
#     val_metric2 = Averager()
#     val_metric3 = Averager()
#     val_metric4 = Averager()
#     val_metric5 = Averager()
#     val_metric6 = Averager()
#
#     pbar = tqdm(loader, leave=False, desc='val')
#     with torch.no_grad():
#         for batch in pbar:
#             img_patch_path, patch_annotation_path = batch["inp_patch_path"], batch["patch_annotation_path"]
#             img_path, img_annotation_path = batch["inp_path"], batch["img_annotation_path"]
#             del batch["inp_patch_path"], batch["gt_patch_path"], batch["patch_annotation_path"], batch["inp_path"], \
#             batch["img_annotation_path"]
#
#             # ----------------------Set img into DocAI------------------------------------------
#             batch_img, batch_img_ratio, batch_img_bbox, batch_text, batch_label = [], [], [], [], []
#             batch_text_tampering_probs, batch_tamper_bbox = [], []
#             batch_patch_original_bbox, batch_patch_norm_bbox, batch_patch_text = [], [], []
#             for j in img_path:
#                 img = Image.open(j)
#                 W, H = img.size
#                 # resize_img = img.resize((target_w, target_h))
#                 # img_name_suffix = j.split('\\')[-1] # 本地
#                 img_name_suffix = j.split('/')[-1]
#                 batch_text_tampering_probs.append(img_text_tampering_probs_dict[img_name_suffix])
#                 # batch_img.append(resize_img)
#                 batch_img.append(img)
#                 batch_img_ratio.append((target_w / W, target_h / H))
#
#             for index, i in enumerate(img_annotation_path):
#                 ratio = batch_img_ratio[index]
#                 with open(i, encoding='utf-8') as r:
#                     img_annotation = json.load(r)
#                 text_list, img_bbox_list, label_list = [], [], []
#                 for item in img_annotation["word_info"]:
#                     word = item["word"]
#                     if len(word) == 0:
#                         continue
#                     if "width" not in item:
#                         x1, y1, x2, y2 = item["left"], item["top"], item["right"], item["bottom"]
#                         w, h = (x2 - x1), (y2 - y1)
#                     else:
#                         x1, y1, w, h = item["left"], item["top"], item["width"], item["height"]
#                         x2, y2 = x1 + w, y1 + h
#                     if "tamper" not in item:
#                         label = 0
#                     else:
#                         label = item["tamper"]
#
#                     resized_x1, resized_y1, resized_x2, resized_y2 = int(x1 * ratio[0]), int(y1 * ratio[1]), int(x2 * ratio[0]), int(y2 * ratio[1])
#                     resized_x1, resized_y1, resized_x2, resized_y2 = Corrdinate_correction(resized_x1, resized_y1, resized_x2, resized_y2)
#                     img_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])
#                     text_list.append(word)
#                     label_list.append(label)
#
#                 batch_img_bbox.append(img_bbox_list)
#                 batch_text.append(text_list)
#                 batch_label.append(label_list)
#
#             batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label, max_length=1024,
#                                        padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
#             offset_mapping = batch_encoding.pop('offset_mapping')  # 返回字符偏移量映射表，长度等于文本段数目的列表，列表中每一项是一个长度等于段中字符数目的整数列表，用于表示每一个字符在原始文本中的偏移量，例如"Teacher" -> "GTE", "ACH","ER"
#
#             bbox, attention_mask = batch_encoding["bbox"].to(device), batch_encoding["attention_mask"].to(device)
#             # Process img_patch
#             for index, i in enumerate(patch_annotation_path):
#                 ratio = batch_img_ratio[index]
#                 with open(i, encoding='utf-8') as r:
#                     patch_annotation = json.load(r)
#                 patch_original_bbox_list, patch_norm_bbox_list, patch_text_list = [], [], []
#                 tamper_bbox_list = []
#                 for item in patch_annotation["word_info"]:
#                     word = item["word"]
#                     if len(word) == 0:
#                         continue
#
#                     x1, y1, x2, y2 = item["Original_left"], item["Original_top"], item["Original_right"], item["Original_bottom"]
#                     resized_x1, resized_y1, resized_x2, resized_y2 = int(x1 * ratio[0]), int(y1 * ratio[1]), int(x2 * ratio[0]), int(y2 * ratio[1])
#                     resized_x1, resized_y1, resized_x2, resized_y2 = Corrdinate_correction(resized_x1, resized_y1,resized_x2, resized_y2)
#                     patch_original_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])
#
#                     patch_x1, patch_y1, patch_x2, patch_y2 = item["left"], item["top"], item["right"], item["bottom"]
#                     norm_patch_x1, norm_patch_y1, norm_patch_x2, norm_patch_y2 = patch_x1 / target_w, patch_y1 / target_h, \
#                                                                                  patch_x2 / target_w, patch_y2 / target_h
#                     patch_norm_bbox_list.append([norm_patch_x1, norm_patch_y1, norm_patch_x2, norm_patch_y2])
#                     patch_text_list.append(word)
#                     if "tamper" not in item:
#                         continue
#                     else:
#                         if item["tamper"] == 1:  # 保存篡改区域的坐标信息
#                             patch_tamper_x1, patch_tamper_y1, patch_tamper_x2, patch_tamper_y2 = item["left"], item["top"], item["right"], item["bottom"]
#                             norm_tamper_x1, norm_tamper_y1, norm_tamper_x2, norm_tamper_y2 = patch_tamper_x1 / target_w, patch_tamper_y1 / target_h, \
#                                                                                              patch_tamper_x2 / target_w, patch_tamper_y2 / target_h
#                             tamper_bbox_list.append([norm_tamper_x1, norm_tamper_y1, norm_tamper_x2, norm_tamper_y2])
#
#                 batch_patch_original_bbox.append(patch_original_bbox_list)
#                 batch_patch_norm_bbox.append(patch_norm_bbox_list)  # 文本位置
#                 batch_patch_text.append(patch_text_list)
#                 batch_tamper_bbox.append(tamper_bbox_list)  # 篡改文本的位置
#
#             max_length = max(len(sub_l) for sub_l in batch_tamper_bbox)
#             batch_tamper_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length - len(sub_l)) for sub_l in batch_tamper_bbox]  # Padding
#             visual_labels = torch.tensor(batch_tamper_bbox_padding, dtype=torch.float).to(device)
#
#             max_length1 = max(len(sub_l) for sub_l in batch_patch_norm_bbox)
#             batch_patch_norm_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length1 - len(sub_l)) for sub_l in batch_patch_norm_bbox]  # Padding
#             batch_patch_norm_bbox_padding = torch.tensor(batch_patch_norm_bbox_padding, dtype=torch.float).to(device)
#
#             max_length3 = max(len(sub_l) for sub_l in batch_text_tampering_probs)
#             max_length2 = attention_mask.shape[1]
#             max_length = max(max_length2, max_length3)
#             batch_text_tampering_probs_padding = [sub_l + [0.] * (max_length - len(sub_l)) for sub_l in batch_text_tampering_probs]  # Padding
#             batch_text_tampering_probs_padding = torch.tensor(batch_text_tampering_probs_padding, dtype=torch.float).unsqueeze(-1).to(device)  # (B, token_length, 1)
#
#             max_length4 = max(len(sub_l) for sub_l in batch_patch_original_bbox)
#             batch_patch_original_bbox_padding = [sub_l + [[0, 0, 0, 0]] * (max_length4 - len(sub_l)) for sub_l in batch_patch_original_bbox]  # Padding
#             batch_patch_original_bbox_padding = torch.tensor(batch_patch_original_bbox_padding, dtype=torch.long).to(device)
#
#             for k, v in batch.items():
#                 batch[k] = v.to(device)
#
#             inp = batch['inp_patch']
#
#             pred_mask = multimodal.infer(inp, bbox=bbox, attention_mask=attention_mask,
#                                          patch_norm_bbox=batch_patch_norm_bbox_padding,
#                                          patch_original_bbox=batch_patch_original_bbox_padding,
#                                          offset_mapping=offset_mapping,
#                                          text_tampering_prob=batch_text_tampering_probs_padding)[0]
#             pred = torch.sigmoid(pred_mask)
#
#             pred[pred >= 0.5] = 1
#             pred[pred < 0.5] = 0
#
#             # pred = torch.sigmoid(model.infer(inp))
#             result1, result2, result3, result4, result5, result6 = metric_fn(pred, batch['gt_patch'])
#             val_metric1.add(result1.item(), inp.shape[0])
#             val_metric2.add(result2.item(), inp.shape[0])
#             val_metric3.add(result3.item(), inp.shape[0])
#             val_metric4.add(result4.item(), inp.shape[0])
#             val_metric5.add(result5.item(), inp.shape[0])
#             val_metric6.add(result6.item(), inp.shape[0])
#
#     # 验证集上的平均指标
#     return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_metric5.item(), val_metric6.item(),\
#            metric1, metric2, metric3, metric4, metric5, metric6
#
#
# def prepare_training():
#     if config.get('resume') is not None:
#         model = models.make(config['model']).cuda()
#         optimizer = utils.make_optimizer(
#             model.parameters(), config['optimizer'])
#         epoch_start = config.get('resume') + 1
#     else:
#         model = models.make(config['model']).cuda()
#         optimizer = utils.make_optimizer(
#             model.parameters(), config['optimizer'])
#         epoch_start = 1
#     max_epoch = config.get('epoch_max')
#     lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
#     log('visual_model: #params={}'.format(utils.compute_num_params(model, text=True)))
#
#     return model, optimizer, epoch_start, lr_scheduler
#
# def train(train_loader, multimodal, processor, img_text_tampering_probs_dict): # 读取CSV中存储的文本篡改概率
#     multimodal.train()
#     loss_list = []
#
#     train_loss_G = utils.Averager()
#     target_w, target_h = config['data_size'], config['data_size']
#
#     for batch in tqdm(train_loader, leave=False, desc='train'):
#         img_patch_path, patch_annotation_path = batch["inp_patch_path"], batch["patch_annotation_path"]
#         img_path, img_annotation_path = batch["inp_path"], batch["img_annotation_path"]
#         del batch["inp_patch_path"], batch["gt_patch_path"], batch["patch_annotation_path"], batch["inp_path"], batch["img_annotation_path"]
#         print("img_patch_path", img_patch_path)
#         print("img_path", img_path)
#
#         #----------------------Set img into DocAI------------------------------------------
#         batch_img, batch_img_ratio, batch_img_bbox, batch_text, batch_label = [], [], [], [], []
#         batch_text_tampering_probs, batch_tamper_bbox = [], []
#         batch_patch_original_bbox, batch_patch_norm_bbox, batch_patch_text = [], [], []
#         for j in img_path:
#             img = Image.open(j)
#             W, H = img.size
#             # resize_img = img.resize((target_w, target_h))
#             # img_name_suffix = j.split('\\')[-1] # 本地
#             img_name_suffix = j.split('/')[-1]
#             batch_text_tampering_probs.append(img_text_tampering_probs_dict[img_name_suffix])
#             # batch_img.append(resize_img)
#             batch_img.append(img)
#             batch_img_ratio.append((target_w / W, target_h / H))
#
#         for index, i in enumerate(img_annotation_path):
#             ratio = batch_img_ratio[index]
#             with open(i, encoding='utf-8') as r:
#                 img_annotation = json.load(r)
#             text_list, img_bbox_list, label_list = [], [], []
#             for item in img_annotation["word_info"]:
#                 word = item["word"]
#                 if len(word) == 0:
#                     continue
#                 if "width" not in item:
#                     x1, y1, x2, y2 = item["left"], item["top"], item["right"], item["bottom"]
#                     w, h = (x2 - x1), (y2 - y1)
#                 else:
#                     x1, y1, w, h = item["left"], item["top"], item["width"], item["height"]
#                     x2, y2 = x1 + w, y1 + h
#
#                 if "tamper" not in item:
#                     label = 0
#                 else:
#                     label = item["tamper"]
#
#                 resized_x1, resized_y1, resized_x2, resized_y2 = int(x1 * ratio[0]), int(y1 * ratio[1]), int(x2 * ratio[0]), int(y2 * ratio[1])
#                 resized_x1, resized_y1, resized_x2, resized_y2 = Corrdinate_correction(resized_x1, resized_y1, resized_x2, resized_y2)
#                 img_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])
#                 text_list.append(word)
#                 label_list.append(label)
#
#             batch_img_bbox.append(img_bbox_list)
#             batch_text.append(text_list)
#             batch_label.append(label_list)
#
#         batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label, max_length=1024,
#                                    padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
#         offset_mapping = batch_encoding.pop('offset_mapping') # 返回字符偏移量映射表，长度等于文本段数目的列表，列表中每一项是一个长度等于段中字符数目的整数列表，用于表示每一个字符在原始文本中的偏移量，例如"Teacher" -> "GTE", "ACH","ER"
#         # is_subword = np.array(offset_mapping[0].tolist())[:, 0] != 0  # 逐个batch处理，根据offset_mapping判断该token是完整的词还是子词
#
#         bbox, attention_mask = batch_encoding["bbox"].to(device), batch_encoding["attention_mask"].to(device)
#         # Process img_patch
#         for index, i in enumerate(patch_annotation_path):
#             ratio = batch_img_ratio[index]
#             with open(i, encoding='utf-8') as r:
#                 patch_annotation = json.load(r)
#             patch_original_bbox_list, patch_norm_bbox_list, patch_text_list = [], [], []
#             tamper_bbox_list = []
#             for item in patch_annotation["word_info"]:
#                 word = item["word"]
#                 if len(word) == 0:
#                     continue
#
#                 x1, y1, x2, y2 = item["Original_left"], item["Original_top"], item["Original_right"], item["Original_bottom"]
#                 resized_x1, resized_y1, resized_x2, resized_y2 = int(x1 * ratio[0]), int(y1 * ratio[1]), int(x2 * ratio[0]), int(y2 * ratio[1])
#                 resized_x1, resized_y1, resized_x2, resized_y2 = Corrdinate_correction(resized_x1, resized_y1, resized_x2, resized_y2)
#                 patch_original_bbox_list.append([resized_x1, resized_y1, resized_x2, resized_y2])
#
#                 patch_x1, patch_y1, patch_x2, patch_y2 = item["left"], item["top"], item["right"], item["bottom"]
#                 norm_patch_x1, norm_patch_y1, norm_patch_x2, norm_patch_y2 = patch_x1 / target_w, patch_y1 / target_h, \
#                                                                              patch_x2 / target_w, patch_y2 / target_h
#                 patch_norm_bbox_list.append([norm_patch_x1, norm_patch_y1, norm_patch_x2, norm_patch_y2])
#                 patch_text_list.append(word)
#                 # if "tamper" not in item:
#                 #     continue
#                 # else:
#                 #     if item["tamper"] == 1: # 保存篡改区域的坐标信息
#                 #         patch_tamper_x1, patch_tamper_y1, patch_tamper_x2, patch_tamper_y2 = item["left"], item["top"], item["right"], item["bottom"]
#                 #         norm_tamper_x1, norm_tamper_y1, norm_tamper_x2, norm_tamper_y2 = patch_tamper_x1 / target_w, patch_tamper_y1 / target_h, \
#                 #                                                                          patch_tamper_x2 / target_w, patch_tamper_y2 / target_h
#                 #         tamper_bbox_list.append([norm_tamper_x1, norm_tamper_y1, norm_tamper_x2, norm_tamper_y2])
#
#             batch_patch_original_bbox.append(patch_original_bbox_list)
#             batch_patch_norm_bbox.append(patch_norm_bbox_list) # 文本位置
#             batch_patch_text.append(patch_text_list)
#             # batch_tamper_bbox.append(tamper_bbox_list) # 篡改文本的位置
#
#         # max_length = max(len(sub_l) for sub_l in batch_tamper_bbox)
#         # batch_tamper_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length - len(sub_l)) for sub_l in batch_tamper_bbox]  # Padding
#         # visual_labels = torch.tensor(batch_tamper_bbox_padding, dtype=torch.float).to(device)
#
#         max_length1 = max(len(sub_l) for sub_l in batch_patch_norm_bbox)
#         batch_patch_norm_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length1 - len(sub_l)) for sub_l in batch_patch_norm_bbox]  # Padding
#         batch_patch_norm_bbox_padding = torch.tensor(batch_patch_norm_bbox_padding, dtype=torch.float).to(device)
#
#         max_length2 = attention_mask.shape[1]
#         max_length3 = max(len(sub_l) for sub_l in batch_text_tampering_probs)
#         max_length = max(max_length2, max_length3)
#         batch_text_tampering_probs_padding = [sub_l + [0.] * (max_length - len(sub_l)) for sub_l in batch_text_tampering_probs]  # Padding
#         batch_text_tampering_probs_padding = torch.tensor(batch_text_tampering_probs_padding, dtype=torch.float).unsqueeze(-1).to(device) # (B, token_length, 1)
#
#         max_length4 = max(len(sub_l) for sub_l in batch_patch_original_bbox)
#         batch_patch_original_bbox_padding = [sub_l + [[0, 0, 0, 0]] * (max_length4 - len(sub_l)) for sub_l in batch_patch_original_bbox]  # Padding
#         batch_patch_original_bbox_padding = torch.tensor(batch_patch_original_bbox_padding, dtype=torch.long).to(device)
#
#
#         for k, v in batch.items():
#             batch[k] = v.to(device)
#         inp_patch = batch['inp_patch']
#         gt_patch = batch['gt_patch']
#
#         multimodal.visual_set_input(inp_patch, gt_patch)
#         multimodal.optimize_parameters(bbox=bbox, attention_mask=attention_mask, patch_norm_bbox=batch_patch_norm_bbox_padding, patch_original_bbox=batch_patch_original_bbox_padding,
#                                        offset_mapping=offset_mapping, text_tampering_probs=batch_text_tampering_probs_padding)
#
#         batch_loss = [torch.zeros_like(multimodal.loss_G) for _ in range(dist.get_world_size())]
#         dist.all_gather(batch_loss, multimodal.loss_G)
#         loss_list.extend(batch_loss)
#
#         loss = [i.item() for i in loss_list]
#
#     return mean(loss)
#
#
# def main(config_, save_path, args):
#     global config, log, writer, log_info
#     config = config_
#     log, writer = utils.set_save_path(save_path, remove=False)
#     with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
#         yaml.dump(config, f, sort_keys=False)
#
#     train_loader, val_loader = make_data_loaders()
#     if config.get('data_norm') is None:
#         config['data_norm'] = {
#             'inp': {'sub': [0], 'div': [1]},
#             'gt': {'sub': [0], 'div': [1]}
#         }
#
#     model, optimizer, epoch_start, lr_scheduler = prepare_training()
#     sam_checkpoint = torch.load(config['sam_checkpoint'], map_location=device)
#     model.load_state_dict(sam_checkpoint, strict=False)
#     multimodal = Visual_DocAI_Net(model, device=device, channel_attention=config['channel_attention'])
#     multimodal.optimizer = optimizer
#     lr_scheduler = CosineAnnealingLR(multimodal.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))
#
#     multimodal = multimodal.cuda()
#     multimodal = torch.nn.parallel.DistributedDataParallel(
#         multimodal,
#         device_ids=[args.local_rank],
#         output_device=args.local_rank,
#         find_unused_parameters=True,
#         broadcast_buffers=False
#     )
#     multimodal = multimodal.module
#
#     if config['DocAI'] == 'Layoutlmv3':
#         processor = AutoProcessor.from_pretrained("./DocAI_pretrained/Layoutlmv3", apply_ocr=False)
#     elif config['DocAI'] == 'Layoutlmv2':
#         processor = AutoProcessor.from_pretrained("./DocAI_pretrained/Layoutlmv2", apply_ocr=False)
#     elif config['DocAI'] == 'Lilt':
#         processor = LayoutLMv3Processor.from_pretrained("./DocAI_pretrained/Lilt-funsd", apply_ocr=False)
#
#     name_list = []
#     for name, para in multimodal.named_parameters():
#         name_list.append(name)
#         if "image_encoder" in name and "prompt_generator" not in name:
#             if "Adapter" in name:
#                 para.requires_grad_(True)
#             else:
#                 para.requires_grad_(False)
#         if "mask_decoder" in name:
#             para.requires_grad_(True)
#
#     print(name_list)
#     model_total_params = sum(p.numel() for p in multimodal.parameters())
#     model_grad_params = sum(p.numel() for p in multimodal.parameters() if p.requires_grad)
#     model_grad_name = [name for name, param in multimodal.named_parameters() if param.requires_grad]
#     print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
#     print(model_grad_name)
#
#     epoch_max = config['epoch_max']
#     epoch_val = config.get('epoch_val')
#     max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
#
#     csv_path = config['csv_path']
#     img_text_tampering_probs_dict = {}
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         f_csv = csv.reader(f)
#         header = next(f_csv)
#         for row in f_csv:
#             img_text_tampering_probs_dict[row[0]] = ast.literal_eval(row[1])
#
#     timer = utils.Timer()
#     for epoch in range(epoch_start, epoch_max + 1):
#         t_epoch_start = timer.t()
#         train_loss_G = train(train_loader, multimodal, processor, img_text_tampering_probs_dict)
#         lr_scheduler.step()
#
#         log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
#         writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
#         log_info.append('train G: loss={:.4f}'.format(train_loss_G))
#         writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
#
#         optimizer_spec = config['optimizer']
#         optimizer_spec['sd'] = optimizer.state_dict()
#
#         # save(config, model, save_path, 'last')
#         if epoch % 5 == 0:
#             save(config, multimodal, save_path, str(epoch))
#
#         if (epoch_val is not None) and (epoch % epoch_val == 0):
#             result1, result2, result3, result4, result5, result6, \
#             metric1, metric2, metric3, metric4, metric5, metric6 = eval_psnr(val_loader, multimodal, processor, img_text_tampering_probs_dict,
#                                                                              eval_type=config.get('eval_type'))
#
#             log_info.append('val: {}={:.4f}'.format(metric1, result1))
#             writer.add_scalars(metric1, {'val': result1}, epoch)
#             log_info.append('val: {}={:.4f}'.format(metric2, result2))
#             writer.add_scalars(metric2, {'val': result2}, epoch)
#             log_info.append('val: {}={:.4f}'.format(metric3, result3))
#             writer.add_scalars(metric3, {'val': result3}, epoch)
#             log_info.append('val: {}={:.4f}'.format(metric4, result4))
#             writer.add_scalars(metric4, {'val': result4}, epoch)
#             log_info.append('val: {}={:.4f}'.format(metric5, result5))
#             writer.add_scalars(metric5, {'val': result5}, epoch)
#             log_info.append('val: {}={:.4f}'.format(metric6, result6))
#             writer.add_scalars(metric6, {'val': result6}, epoch)
#
#             if config['eval_type'] != 'ber': # 选择IoU
#                 if result3 > max_val_v:
#                     max_val_v = result3
#                     save(config, multimodal, save_path, '{}_best'.format(str(epoch)))
#             else:
#                 if result3 < max_val_v:
#                     max_val_v = result3
#                     save(config, multimodal, save_path, '{}_best'.format(str(epoch)))
#
#             t = timer.t()
#             prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
#             t_epoch = utils.time_text(t - t_epoch_start)
#             t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
#             log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
#
#             log(', '.join(log_info))
#             writer.flush()
#
#
# def save(config, multimodal, save_path, name):
#     if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
#         if config['model']['args']['encoder_mode']['name'] == 'evp':
#             prompt_generator = multimodal.encoder.backbone.prompt_generator.state_dict()
#             decode_head = multimodal.encoder.decode_head.state_dict()
#             torch.save({"prompt": prompt_generator, "decode_head": decode_head},
#                        os.path.join(save_path, f"prompt_epoch_{name}.pth"))
#         else:
#             torch.save(multimodal.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
#     else:
#         torch.save(multimodal.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
#     parser.add_argument('--name', default=None)
#     parser.add_argument('--tag', default=None)
#     parser.add_argument("--local_rank", type=int, default=-1, help="")
#     args = parser.parse_args()
#
#     with open(args.config, 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#         print('config loaded.')
#
#     save_name = args.name
#     if save_name is None:
#         save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
#     save_path = os.path.join('./save', save_name)
#
#     main(config, save_path, args=args)



# DocAI输入为Image SAM输入为Patch, 边界框归一化到0-1000
import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import copy
import json
import csv
import ast
from PIL import Image
import torch.distributed as dist
from models.multimodal_skip import Visual_DocAI_Net
from transformers import AutoProcessor, AutoModel, LayoutLMv3Processor, LayoutLMv2Processor

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def normalize_bbox(bbox, width, height): # 归一化边框
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    #     log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, multimodal, processor, img_text_tampering_probs_dict, eval_type=None): # 读取CSV文件中存储的文本篡改概率
    multimodal.eval()
    target_w, target_h = config['data_size'], config['data_size']

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4, metric5, metric6 = 'f1_max', 'auc', 'iou', 'p', 'r', 'f1_mean'

    val_metric1 = Averager()
    val_metric2 = Averager()
    val_metric3 = Averager()
    val_metric4 = Averager()
    val_metric5 = Averager()
    val_metric6 = Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    with torch.no_grad():
        for batch in pbar:
            img_patch_path, patch_annotation_path = batch["inp_patch_path"], batch["patch_annotation_path"]
            img_path, img_annotation_path = batch["inp_path"], batch["img_annotation_path"]
            del batch["inp_patch_path"], batch["gt_patch_path"], batch["patch_annotation_path"], batch["inp_path"], \
            batch["img_annotation_path"]

            # ----------------------Set img into DocAI------------------------------------------
            batch_img, batch_img_WH, batch_img_bbox, batch_text, batch_label = [], [], [], [], []
            batch_text_tampering_probs, batch_tamper_bbox = [], []
            batch_patch_original_bbox, batch_patch_norm_bbox, batch_patch_text = [], [], []
            for j in img_path:
                img = Image.open(j)
                W, H = img.size
                # resize_img = img.resize((target_w, target_h))
                # img_name_suffix = j.split('\\')[-1] # 本地
                img_name_suffix = j.split('/')[-1]
                batch_text_tampering_probs.append(img_text_tampering_probs_dict[img_name_suffix])
                # batch_img.append(resize_img)
                batch_img.append(img)
                batch_img_WH.append((W, H))

                img_annotation_path = j.replace('img', 'annotation').replace('.jpg', '.json')
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

            batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label, max_length=1024,
                                       padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = batch_encoding.pop('offset_mapping')  # 返回字符偏移量映射表，长度等于文本段数目的列表，列表中每一项是一个长度等于段中字符数目的整数列表，用于表示每一个字符在原始文本中的偏移量，例如"Teacher" -> "GTE", "ACH","ER"

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
                batch_patch_norm_bbox.append(patch_norm_bbox_list)  # 文本位置
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

            pred_mask = multimodal.infer(inp, bbox=bbox, attention_mask=attention_mask,
                                         patch_norm_bbox=batch_patch_norm_bbox_padding,
                                         patch_original_bbox=batch_patch_original_bbox_padding,
                                         offset_mapping=offset_mapping,
                                         text_tampering_prob=batch_text_tampering_probs_padding)[0]
            pred = torch.sigmoid(pred_mask)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            # pred = torch.sigmoid(model.infer(inp))
            result1, result2, result3, result4, result5, result6 = metric_fn(pred, batch['gt_patch'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])
            val_metric5.add(result5.item(), inp.shape[0])
            val_metric6.add(result6.item(), inp.shape[0])

    # 验证集上的平均指标
    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_metric5.item(), val_metric6.item(),\
           metric1, metric2, metric3, metric4, metric5, metric6


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    log('visual_model: #params={}'.format(utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, multimodal, processor, img_text_tampering_probs_dict): # 读取CSV中存储的文本篡改概率
    multimodal.train()
    loss_list = []

    train_loss_G = utils.Averager()
    target_w, target_h = config['data_size'], config['data_size']

    for batch in tqdm(train_loader, leave=False, desc='train'):

        img_patch_path, patch_annotation_path = batch["inp_patch_path"], batch["patch_annotation_path"]
        img_path, img_annotation_path = batch["inp_path"], batch["img_annotation_path"]
        del batch["inp_patch_path"], batch["gt_patch_path"], batch["patch_annotation_path"], batch["inp_path"], batch["img_annotation_path"]
        print("img_patch_path", img_patch_path)
        print("img_path", img_path)

        #----------------------Set img into DocAI------------------------------------------
        batch_img, batch_img_WH, batch_img_bbox, batch_text, batch_label = [], [], [], [], []
        batch_text_tampering_probs, batch_tamper_bbox = [], []
        batch_patch_original_bbox, batch_patch_norm_bbox, batch_patch_text = [], [], []
        for j in img_path:
            img = Image.open(j)
            W, H = img.size
            # resize_img = img.resize((target_w, target_h))
            # img_name_suffix = j.split('\\')[-1] # 本地
            img_name_suffix = j.split('/')[-1]
            batch_text_tampering_probs.append(img_text_tampering_probs_dict[img_name_suffix])
            # batch_img.append(resize_img)
            batch_img.append(img)
            batch_img_WH.append((W, H))

            img_annotation_path = j.replace('img', 'annotation').replace('.jpg', '.json')

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

        batch_encoding = processor(images=batch_img, text=batch_text, boxes=batch_img_bbox, word_labels=batch_label, max_length=1024,
                                   padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = batch_encoding.pop('offset_mapping') # 返回字符偏移量映射表，长度等于文本段数目的列表，列表中每一项是一个长度等于段中字符数目的整数列表，用于表示每一个字符在原始文本中的偏移量，例如"Teacher" -> "GTE", "ACH","ER"
        # is_subword = np.array(offset_mapping[0].tolist())[:, 0] != 0  # 逐个batch处理，根据offset_mapping判断该token是完整的词还是子词

        bbox, attention_mask = batch_encoding["bbox"].to(device), batch_encoding["attention_mask"].to(device)
        # Process img_patch
        for index, i in enumerate(patch_annotation_path):
            W, H = batch_img_WH[index]
            with open(i, encoding='utf-8') as r:
                patch_annotation = json.load(r)
            patch_original_bbox_list, patch_norm_bbox_list, patch_text_list = [], [], []
            tamper_bbox_list = []
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
                # if "tamper" not in item:
                #     continue
                # else:
                #     if item["tamper"] == 1: # 保存篡改区域的坐标信息
                #         patch_tamper_x1, patch_tamper_y1, patch_tamper_x2, patch_tamper_y2 = item["left"], item["top"], item["right"], item["bottom"]
                #         norm_tamper_x1, norm_tamper_y1, norm_tamper_x2, norm_tamper_y2 = patch_tamper_x1 / target_w, patch_tamper_y1 / target_h, \
                #                                                                          patch_tamper_x2 / target_w, patch_tamper_y2 / target_h
                #         tamper_bbox_list.append([norm_tamper_x1, norm_tamper_y1, norm_tamper_x2, norm_tamper_y2])

            batch_patch_original_bbox.append(patch_original_bbox_list)
            batch_patch_norm_bbox.append(patch_norm_bbox_list) # 文本位置
            batch_patch_text.append(patch_text_list)
            # batch_tamper_bbox.append(tamper_bbox_list) # 篡改文本的位置

        # max_length = max(len(sub_l) for sub_l in batch_tamper_bbox)
        # batch_tamper_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length - len(sub_l)) for sub_l in batch_tamper_bbox]  # Padding
        # visual_labels = torch.tensor(batch_tamper_bbox_padding, dtype=torch.float).to(device)

        max_length1 = max(len(sub_l) for sub_l in batch_patch_norm_bbox)
        batch_patch_norm_bbox_padding = [sub_l + [[0., 0., 0., 0.]] * (max_length1 - len(sub_l)) for sub_l in batch_patch_norm_bbox]  # Padding
        batch_patch_norm_bbox_padding = torch.tensor(batch_patch_norm_bbox_padding, dtype=torch.float).to(device)

        max_length2 = attention_mask.shape[1]
        max_length3 = max(len(sub_l) for sub_l in batch_text_tampering_probs)
        max_length = max(max_length2, max_length3)
        batch_text_tampering_probs_padding = [sub_l + [0.] * (max_length - len(sub_l)) for sub_l in batch_text_tampering_probs]  # Padding
        batch_text_tampering_probs_padding = torch.tensor(batch_text_tampering_probs_padding, dtype=torch.float).unsqueeze(-1).to(device) # (B, token_length, 1)

        max_length4 = max(len(sub_l) for sub_l in batch_patch_original_bbox)
        batch_patch_original_bbox_padding = [sub_l + [[0, 0, 0, 0]] * (max_length4 - len(sub_l)) for sub_l in batch_patch_original_bbox]  # Padding
        batch_patch_original_bbox_padding = torch.tensor(batch_patch_original_bbox_padding, dtype=torch.long).to(device)


        for k, v in batch.items():
            batch[k] = v.to(device)
        inp_patch = batch['inp_patch']
        gt_patch = batch['gt_patch']

        multimodal.visual_set_input(inp_patch, gt_patch)
        multimodal.optimize_parameters(bbox=bbox, attention_mask=attention_mask, patch_norm_bbox=batch_patch_norm_bbox_padding, patch_original_bbox=batch_patch_original_bbox_padding,
                                       offset_mapping=offset_mapping, text_tampering_probs=batch_text_tampering_probs_padding)

        batch_loss = [torch.zeros_like(multimodal.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, multimodal.loss_G)
        loss_list.extend(batch_loss)

        loss = [i.item() for i in loss_list]

    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    sam_checkpoint = torch.load(config['sam_checkpoint'], map_location=device)
    model.load_state_dict(sam_checkpoint, strict=False)
    multimodal = Visual_DocAI_Net(model, device=device, channel_attention=config['channel_attention'])
    multimodal.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(multimodal.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    multimodal = multimodal.cuda()
    multimodal = torch.nn.parallel.DistributedDataParallel(
        multimodal,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    multimodal = multimodal.module

    if config['DocAI'] == 'Layoutlmv3':
        processor = AutoProcessor.from_pretrained("./DocAI_pretrained/Layoutlmv3", apply_ocr=False)
    elif config['DocAI'] == 'Layoutlmv2':
        processor = AutoProcessor.from_pretrained("./DocAI_pretrained/Layoutlmv2", apply_ocr=False)
    elif config['DocAI'] == 'Lilt':
        processor = LayoutLMv3Processor.from_pretrained("./DocAI_pretrained/Lilt-funsd", apply_ocr=False)

    name_list = []
    for name, para in multimodal.named_parameters():
        name_list.append(name)
        if "image_encoder" in name and "prompt_generator" not in name:
            if "Adapter" in name:
                para.requires_grad_(True)
            else:
                para.requires_grad_(False)
        if "mask_decoder" in name:
            para.requires_grad_(True)

    print(name_list)
    model_total_params = sum(p.numel() for p in multimodal.parameters())
    model_grad_params = sum(p.numel() for p in multimodal.parameters() if p.requires_grad)
    model_grad_name = [name for name, param in multimodal.named_parameters() if param.requires_grad]
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    print(model_grad_name)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8

    csv_path = config['csv_path']
    img_text_tampering_probs_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        for row in f_csv:
            img_text_tampering_probs_dict[row[0]] = ast.literal_eval(row[1])

    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, multimodal, processor, img_text_tampering_probs_dict)
        lr_scheduler.step()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        # save(config, model, save_path, 'last')
        if epoch % 5 == 0:
            save(config, multimodal, save_path, str(epoch))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, result5, result6, \
            metric1, metric2, metric3, metric4, metric5, metric6 = eval_psnr(val_loader, multimodal, processor, img_text_tampering_probs_dict,
                                                                             eval_type=config.get('eval_type'))

            log_info.append('val: {}={:.4f}'.format(metric1, result1))
            writer.add_scalars(metric1, {'val': result1}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result2))
            writer.add_scalars(metric2, {'val': result2}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric3, result3))
            writer.add_scalars(metric3, {'val': result3}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric4, result4))
            writer.add_scalars(metric4, {'val': result4}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric5, result5))
            writer.add_scalars(metric5, {'val': result5}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric6, result6))
            writer.add_scalars(metric6, {'val': result6}, epoch)

            if config['eval_type'] != 'ber': # 选择IoU
                if result3 > max_val_v:
                    max_val_v = result3
                    save(config, multimodal, save_path, '{}_best'.format(str(epoch)))
            else:
                if result3 < max_val_v:
                    max_val_v = result3
                    save(config, multimodal, save_path, '{}_best'.format(str(epoch)))

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()


def save(config, multimodal, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = multimodal.encoder.backbone.prompt_generator.state_dict()
            decode_head = multimodal.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(multimodal.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(multimodal.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


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