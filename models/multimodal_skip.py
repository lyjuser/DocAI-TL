import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, LayoutLMv3ForTokenClassification
from torch.nn import CrossEntropyLoss
from typing import Any, Optional, Tuple
from models.loss import _iou_loss, BBCEWithLogitLoss, IOU
from .dice import DiceLoss
from .soft_bce import SoftBCEWithLogitsLoss
from models.MultLoss import WeightedDiceBCE

def dice_loss(pred, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""
    pred = torch.sigmoid(pred)
    batch_size = pred.shape[0]
    smooth = 1e-5

    i_flat = pred.view(batch_size, -1)
    t_flat = target.view(batch_size, -1)

    intersection = (i_flat * t_flat).sum(-1)
    union = (i_flat * i_flat).sum(-1) + (t_flat * t_flat).sum(-1)
    dice = 1 - ((2 * intersection + smooth) / (union + smooth))
    loss = dice.mean()
    return loss

class ChannelAttention(nn.Module):
    def __init__(self, channel=256, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xs = x
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        # output = x + x * output
        output = x * output
        return output

class Feature_refine(nn.Module):
    def __init__(self, channel=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channel, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Visual_DocAI_Net(nn.Module):
    def __init__(self, model, device, channel_attention=False):
        super(Visual_DocAI_Net, self).__init__()
        self.model = model
        self.device = device
        self.channel_attention = channel_attention
        if channel_attention:
            self.CAM = ChannelAttention()
            self.FRE = Feature_refine()

        self.inp_size = self.model.image_encoder.img_size
        self.patch_size = self.model.image_encoder.patch_size
        self.image_embedding_size = self.inp_size // 16

        self.criterionBCE = BBCEWithLogitLoss()
        # self.criterionBCE = SoftBCEWithLogitsLoss(smooth_factor=0.1)
        self.criterionDice = DiceLoss(mode='binary')
        self.criterion = WeightedDiceBCE(dice_weight=0.3, BCE_weight=0.7)

    def visual_set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0.3 * self.criterionBCE(self.pred_mask, self.gt_mask)
        self.loss_G += 0.7 * _iou_loss(self.pred_mask, self.gt_mask)  # bce and iou ratio? 1:1/3:7?

        # self.loss_G += 0.7 * dice_loss(self.pred_mask, self.gt_mask) # bce and dice ratio? 1:1/3:7?
        # self.loss_G += 0.7 * self.criterionDice(self.pred_mask, self.gt_mask)  # bce and dice ratio? 1:1/3:7?

        # pred_mask = torch.sigmoid(self.pred_mask)
        # self.loss_G = self.criterion(pred_mask, self.gt_mask) # bce and dice ratio? 1:1/3:7?
        self.loss_G.backward()

    def optimize_parameters(self, bbox=None, attention_mask=None, patch_norm_bbox=None, patch_original_bbox=None, offset_mapping=None, text_tampering_probs=None):
        self.forward(bbox, attention_mask, patch_norm_bbox, patch_original_bbox, offset_mapping, text_tampering_probs)
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.model.pe_layer(self.image_embedding_size).unsqueeze(0)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.inp_size, self.inp_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def probability_map_generation(self, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
                                   patch_original_bbox, offset_mapping, num_patches=14):
        B, L = patch_norm_bbox.shape[0], num_patches
        ocr_points_x1 = torch.clip(torch.floor(patch_norm_bbox[:, :, 0] * L).long(), 0, L - 1)
        ocr_points_x2 = torch.clip(torch.floor(patch_norm_bbox[:, :, 2] * L).long(), 0, L - 1)

        ocr_points_y1 = torch.clip(torch.floor(patch_norm_bbox[:, :, 1] * L).long(), 0, L - 1) * L
        ocr_points_y2 = torch.clip(torch.floor(patch_norm_bbox[:, :, 3] * L).long(), 0, L - 1) * L
        ocr_points_left_top = ocr_points_x1 + ocr_points_y1
        ocr_points_right_top = ocr_points_x2 + ocr_points_y1
        ocr_points_left_bottom = ocr_points_x1 + ocr_points_y2
        ocr_points_right_bottom = ocr_points_x2 + ocr_points_y2

        gap_y = ((ocr_points_left_bottom - ocr_points_left_top) // L).tolist()
        probability_map = torch.zeros(B, 4096, dtype=torch.float)
        for i in range(B):  # 逐个Batch处理
            num_nonzero = torch.sum(attention_mask[i] != 0).item() # 实际token长度，去除Pad
            is_subword = np.array(offset_mapping[i][:num_nonzero].tolist())[:, 0] != 0  # 逐个batch处理，根据offset_mapping判断该token是完整的词还是子词
            # print("is_subword:", is_subword)
            # print('len:', len(is_subword))
            # a = text_tampering_prob[i].tolist()
            word_tampering_prob = [pred for idx, pred in enumerate(text_tampering_prob[i][:num_nonzero].tolist()) if not is_subword[idx]]
            word_tampering_prob = torch.tensor(word_tampering_prob).to(self.device)[1:-1] # 去除特殊词元<cls>和<sep>
            # print("nonzero:", num_nonzero)
            # print('bbox nonzero:', bbox[i][:num_nonzero].shape)
            word_bbox = [pred for idx, pred in enumerate(bbox[i][:num_nonzero].tolist()) if not is_subword[idx]]
            word_bbox = torch.tensor(word_bbox).to(self.device)[1:-1]

            index = torch.where((patch_original_bbox[i][:, None] == word_bbox).all(-1))[1].tolist() # 从整张Image中取出对应Patch的文本篡改概率
            patch_word_tampering_prob = word_tampering_prob[index]
            # 遍历 patch_original_bbox 中的每个值，检查是否存在于 word_bbox 中
            matching_indices = []
            for idx, patch_box in enumerate(patch_original_bbox[i]):
                # 检查当前 patch_box 是否在 word_bbox 中,在则记录值在patch_box中的位置，主要对付Layoutlmv2
                if any(torch.equal(patch_box, word_box) for word_box in word_bbox):
                    matching_indices.append(idx)

            # # print("patch_norm_bbox", patch_norm_bbox[i])
            # print("patch_original_bbox", patch_original_bbox[i])
            # # print('bbox:', bbox[i][:num_nonzero].detach().cpu().numpy())
            # # print('len bbox:', len(bbox[i][:num_nonzero].detach().cpu().numpy()))
            # print("word_bbox", word_bbox)
            # print("index:", index)
            # print("match_index:", matching_indices)
            # # print("patch_word_tampering_prob shape:", patch_word_tampering_prob.shape)
            # print("gay_y:", gap_y[i])
            # # print("gay_y length:", len(gap_y[i]))

            gap_y_indics = torch.tensor(gap_y[i]).to(self.device)[matching_indices].tolist()
            all_positions, all_tampering_probs = [], [] # 加跳跃连接
            # for j, value in enumerate(gap_y[i]): # 逐个文本区域处理
            for j, value in enumerate(gap_y_indics):  # 逐个文本区域处理
                for id in range(value + 1):
                    left_top = ocr_points_left_top[i, j].item()
                    right_top = ocr_points_right_top[i, j].item()
                    if left_top == 0 and right_top == 0:
                        continue
                    if right_top - left_top < 0:
                        continue
                    index_left = left_top + (id * L)
                    index_right = right_top + (id * L)
                    position = np.arange(index_left, index_right + 1, dtype=np.longlong)
                    # if patch_word_tampering_prob[j].item() >= 0.5:
                    #     tampering_prob = np.full(index_right - index_left + 1, 1.0, dtype=np.float32)
                    # else:
                    #     tampering_prob = np.full(index_right - index_left + 1, 0.0, dtype=np.float32)
                    tampering_prob = np.full(index_right-index_left+1, patch_word_tampering_prob[j].item(), dtype=np.float32)
                    if len(all_positions) == 0:
                        all_positions.append(position)
                        all_tampering_probs.append(tampering_prob)
                    else:
                        all_positions[0] = np.append(all_positions[0], position, axis=0)
                        all_tampering_probs[0] = np.append(all_tampering_probs[0], tampering_prob, axis=0)

            probability_map[i][all_positions[0]] = torch.from_numpy(all_tampering_probs[0])
        probability_map = probability_map.unsqueeze(-1).reshape(B, L, L).unsqueeze(1).to(self.device) # (B, 1, 64, 64)
        return probability_map

    def size_probability_map_generation(self, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
                                        patch_original_bbox, offset_mapping, size):
        threshold = 0.5
        B = patch_norm_bbox.shape[0]
        batch_probability_maps = []
        for i in range(B):  # 逐个Batch处理
            num_nonzero = torch.sum(attention_mask[i] != 0).item() # 实际token长度，去除Padding
            is_subword = np.array(offset_mapping[i][:num_nonzero].tolist())[:, 0] != 0  # 逐个batch处理，根据offset_mapping判断该token是完整的词还是子词
            # a = text_tampering_prob[i].tolist()
            word_tampering_prob = [pred for idx, pred in enumerate(text_tampering_prob[i][:num_nonzero].tolist()) if not is_subword[idx]]
            word_tampering_prob = torch.tensor(word_tampering_prob).to(self.device)[1:-1] # 去除特殊词元<cls>和<sep>
            word_bbox = [pred for idx, pred in enumerate(bbox[i][:num_nonzero].tolist()) if not is_subword[idx]]
            word_bbox = torch.tensor(word_bbox).to(self.device)[1:-1]

            index = torch.where((patch_original_bbox[i][:, None] == word_bbox).all(-1))[1].tolist() # 从整张Image中取出对应Patch的文本篡改概率
            patch_word_tampering_prob = word_tampering_prob[index]
            probability_map = np.zeros((1, size, size))
            size_bbox, tampering_probs_list = [list(map(lambda y: int(y * size), x)) for x in patch_norm_bbox[i].cpu().tolist() if x != [0., 0., 0., 0.]], \
                                              patch_word_tampering_prob.squeeze(-1).cpu().tolist()

            for idx, item in enumerate(size_bbox):
                probability_map[0, item[1]:item[3]+1, item[0]:item[2]+1] = tampering_probs_list[idx]
                if tampering_probs_list[idx] >= threshold:
                    probability_map[0, item[1]:item[3] + 1, item[0]:item[2] + 1] = 1.0
                else:
                    probability_map[0, item[1]:item[3] + 1, item[0]:item[2] + 1] = 0.0
            batch_probability_maps.append(probability_map)

        batch_probability_maps = torch.tensor(np.array(batch_probability_maps), dtype=torch.float).to(self.device)
        return batch_probability_maps

    def forward(self, bbox=None, attention_mask=None, patch_norm_bbox=None, patch_original_bbox=None,
                offset_mapping=None, text_tampering_prob=None):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.model.prompt_embed_dim), device=self.device) # (1, 0, 256)
        dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size) # (1, 256, 64, 64)

        visual_features = self.model.image_encoder(self.input) # (B, 256, 64, 64)
        B, C, H, W = visual_features.shape

        num_patches = self.inp_size // self.patch_size
        probability_map = self.probability_map_generation(text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
                                                          patch_original_bbox, offset_mapping, num_patches)

        # probability_map = self.size_probability_map_generation(text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
        #                                                        patch_original_bbox, offset_mapping, H)

        fusion_features = visual_features + (visual_features * probability_map)
        if self.channel_attention:
            self.fusion_features = self.CAM(fusion_features)
        else:
            self.fusion_features = fusion_features

        # spatial_features = visual_features + (visual_features * probability_map)
        # spatial_features = visual_features * probability_map
        # if self.channel_attention:
        #     channel_features = self.CAM(visual_features)
        #     self.fusion_features = self.FRE(channel_features + spatial_features)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.fusion_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) # (B, 1, 256, 256) -> (B, 1, 1024, 1024)
        self.pred_mask = masks

    # def forward(self, bbox=None, attention_mask=None, patch_norm_bbox=None, patch_original_bbox=None,
    #             offset_mapping=None, text_tampering_prob=None):
    #     bs = 1
    #
    #     # Embed prompts
    #     sparse_embeddings = torch.empty((bs, 0, self.model.prompt_embed_dim), device=self.device) # (1, 0, 256)
    #     # dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
    #     #     bs, -1, self.image_embedding_size, self.image_embedding_size) # (1, 256, 64, 64)
    #
    #     visual_features = self.model.image_encoder(self.input) # (B, 256, 64, 64)
    #     B, C, H, W = visual_features.shape
    #
    #     num_patches = self.inp_size // self.patch_size
    #     probability_map = self.probability_map_generation(visual_features, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
    #                                                       patch_original_bbox, offset_mapping, num_patches)
    #
    #     dense_embeddings = self.model.mask_encoding(probability_map)
    #     # Predict masks
    #     low_res_masks, iou_predictions = self.model.mask_decoder(
    #         image_embeddings=visual_features,
    #         image_pe=self.get_dense_pe(),
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=False,
    #     )
    #
    #     # Upscale the masks to the original image resolution
    #     masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) # (B, 1, 256, 256) -> (B, 1, 1024, 1024)
    #     self.pred_mask = masks

    def infer(self, input, bbox=None, attention_mask=None, patch_norm_bbox=None, patch_original_bbox=None,
              offset_mapping=None, text_tampering_prob=None):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.model.prompt_embed_dim), device=self.device) # (1, 0, 256)
        dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size) # (1, 256, 64, 64)

        visual_features = self.model.image_encoder(input) # (B, 256, 64, 64)
        B, C, H, W = visual_features.shape

        num_patches = self.inp_size // self.patch_size
        probability_map = self.probability_map_generation(text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
                                                          patch_original_bbox, offset_mapping, num_patches)

        # probability_map = self.size_probability_map_generation(text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
        #                                                        patch_original_bbox, offset_mapping, H)

        fusion_features = visual_features + (visual_features * probability_map)
        if self.channel_attention:
            self.fusion_features = self.CAM(fusion_features)
        else:
            self.fusion_features = fusion_features

        # spatial_features = visual_features * probability_map
        # # spatial_features = visual_features + (visual_features * probability_map)
        # if self.channel_attention:
        #     channel_features = self.CAM(visual_features)
        #     self.fusion_features = self.FRE(channel_features + spatial_features)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.fusion_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) # (B, 1, 256, 256) -> (B, 1, 1024, 1024)
        probability_map = self.postprocess_masks(probability_map, self.inp_size, self.inp_size)
        return masks, probability_map

    # def infer(self, input, bbox=None, attention_mask=None, patch_norm_bbox=None, patch_original_bbox=None,
    #           offset_mapping=None, text_tampering_prob=None):
    #     bs = 1
    #
    #     # Embed prompts
    #     sparse_embeddings = torch.empty((bs, 0, self.model.prompt_embed_dim), device=self.device) # (1, 0, 256)
    #     dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
    #         bs, -1, self.image_embedding_size, self.image_embedding_size) # (1, 256, 64, 64)
    #
    #     self.visual_features = self.model.image_encoder(input) # (B, 256, 64, 64)
    #     B, C, H, W = self.visual_features.shape
    #
    #     num_patches = self.inp_size // self.patch_size
    #     probability_map = self.probability_map_generation(self.visual_features, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
    #                                                       patch_original_bbox, offset_mapping, num_patches)
    #     # probability_map = self.probability_map_generation_1024(text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
    #     #                                                        patch_original_bbox, offset_mapping)
    #
    #     dense_embeddings = self.model.mask_encoding(probability_map)
    #     # Predict masks
    #     low_res_masks, iou_predictions = self.model.mask_decoder(
    #         image_embeddings=self.visual_features,
    #         image_pe=self.get_dense_pe(),
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=False,
    #     )
    #     # probability_map = F.interpolate(probability_map, (256, 256), mode="bilinear", align_corners=False)
    #     # low_res_masks = low_res_masks * probability_map
    #
    #     # Upscale the masks to the original image resolution
    #     masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) # (B, 1, 256, 256) -> (B, 1, 1024, 1024)
    #     # pred = torch.sigmoid(masks)
    #     probability_map = self.postprocess_masks(probability_map, self.inp_size, self.inp_size) # (B, 1, 256, 256) -> (B, 1, 1024, 1024)
    #     # masks = pred * probability_map
    #     return masks, probability_map
