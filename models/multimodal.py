import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, LayoutLMv3ForTokenClassification
from torch.nn import CrossEntropyLoss
from typing import Any, Optional, Tuple
from models.loss import _iou_loss, BBCEWithLogitLoss, IOU

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

class LayoutLMv3TokenTamperingClassification(nn.Module):
    def __init__(self, load_weight=None, num_labels=2):
        super(LayoutLMv3TokenTamperingClassification, self).__init__()
        self.num_labels = num_labels
        self.layoutlmv3 = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
        # self.layoutlmv3 = AutoModel.from_pretrained("/home/lyj/Layoutlmv3 For Token Tampering Classification/pretrained/")
        self.dropout = nn.Dropout(p=0.1)
        if self.num_labels < 10:
            self.classifier = nn.Linear(self.layoutlmv3.config.hidden_size, self.num_labels)

        if load_weight != 'None':
            state_dict = torch.load(load_weight)
            self.layoutlmv3.load_state_dict(state_dict)
        else:
            self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.classifier.weight)
        # nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, input_ids=None, bbox=None, attention_mask=None, pixel_values=None, labels=None):
        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        if input_ids is not None:
            input_shape = input_ids.size()

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 只计算labels != -100处的loss
            selected = (labels != -100)
            logits_selected = logits[selected]
            labels_selected = labels[selected]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits_selected.view(-1, self.num_labels), labels_selected.view(-1))

        return_dict = {}
        return_dict["loss"] = loss
        return_dict["logits"] = logits
        return return_dict

class Visual_DocAI_Net(nn.Module):
    def __init__(self, model, device, channel_attention=False):
        super(Visual_DocAI_Net, self).__init__()
        self.model = model
        self.device = device
        self.inp_size = self.model.image_encoder.img_size
        self.patch_size = self.model.image_encoder.patch_size
        self.image_embedding_size = self.inp_size // 16

        self.criterionBCE = BBCEWithLogitLoss()
        self.criterionCE = CrossEntropyLoss()
        # self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        # self.criterionFL = FocalLoss()

    def visual_set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0.3 * self.criterionBCE(self.pred_mask, self.gt_mask)
        # self.loss_G = self.criterionFL(self.pred_mask, self.gt_mask)
        self.loss_G += 0.7 * _iou_loss(self.pred_mask, self.gt_mask)  # bce and iou ratio? 1:1/3:7?
        # self.loss_G += 0.7 * dice_loss(self.pred_mask, self.gt_mask) # bce and dice ratio? 1:1/3:7?
        self.loss_G.backward()

    def optimize_parameters(self, input_ids=None, bbox=None, attention_mask=None, pixel_values=None, patch_norm_bbox=None, patch_original_bbox=None, visual_labels=None, offset_mapping=None, text_tampering_probs=None):
        self.forward(input_ids, bbox, attention_mask, pixel_values, patch_norm_bbox, patch_original_bbox, visual_labels, offset_mapping, text_tampering_probs)
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

    def generate_label(self, visual_labels):
        B = visual_labels.shape[0]
        L = 14
        ocr_points_x1 = torch.clip(torch.floor(visual_labels[:, :, 0] * L).long(), 0, L - 1)
        ocr_points_x2 = torch.clip(torch.floor(visual_labels[:, :, 2] * L).long(), 0, L - 1)

        ocr_points_y1 = torch.clip(torch.floor(visual_labels[:, :, 1] * L).long(), 0, L - 1) * L
        ocr_points_y2 = torch.clip(torch.floor(visual_labels[:, :, 3] * L).long(), 0, L - 1) * L
        ocr_points_left_top = ocr_points_x1 + ocr_points_y1
        ocr_points_right_top = ocr_points_x2 + ocr_points_y1
        ocr_points_left_bottom = ocr_points_x1 + ocr_points_y2
        ocr_points_right_bottom = ocr_points_x2 + ocr_points_y2

        gap_x = (ocr_points_right_top - ocr_points_left_top).tolist()
        gap_y = ((ocr_points_left_bottom - ocr_points_left_top) // L).tolist()
        ground_truth = np.zeros((B, 4096), dtype=np.int64)

        for i in range(B): # 逐个Batch处理
            all_label = []
            for j, value in enumerate(gap_y[i]): # 逐个篡改区域处理
                for id in range(value + 1):
                    left_top = ocr_points_left_top[i, j].item()
                    right_top = ocr_points_right_top[i, j].item()
                    index_left = left_top + (id * L)
                    index_right = right_top + (id * L)
                    label = np.arange(index_left, index_right + 1, dtype=np.longlong)
                    if len(all_label) == 0:
                        all_label.append(label)
                    else:
                        all_label[0] = np.append(all_label[0], label, axis=0)
            ground_truth[i][all_label[0]] = 1
        ground_truth = torch.tensor(ground_truth)
        return ground_truth

    # def vtfeature_fusion(self, visual_tampering_prob, text_tampering_prob, norm_bbox, offset_mapping, num_patches=14):
    #     B, L = norm_bbox.shape[0], num_patches
    #     ocr_points_x1 = torch.clip(torch.floor(norm_bbox[:, :, 0] * L).long(), 0, L - 1)
    #     ocr_points_x2 = torch.clip(torch.floor(norm_bbox[:, :, 2] * L).long(), 0, L - 1)
    #
    #     ocr_points_y1 = torch.clip(torch.floor(norm_bbox[:, :, 1] * L).long(), 0, L - 1) * L
    #     ocr_points_y2 = torch.clip(torch.floor(norm_bbox[:, :, 3] * L).long(), 0, L - 1) * L
    #     ocr_points_left_top = ocr_points_x1 + ocr_points_y1
    #     ocr_points_right_top = ocr_points_x2 + ocr_points_y1
    #     ocr_points_left_bottom = ocr_points_x1 + ocr_points_y2
    #     ocr_points_right_bottom = ocr_points_x2 + ocr_points_y2
    #
    #     gap_y = ((ocr_points_left_bottom - ocr_points_left_top) // L).tolist()
    #     fusion_prob = visual_tampering_prob.clone()
    #     for i in range(B):  # 逐个Batch处理
    #         all_positions = []
    #         is_subword = np.array(offset_mapping[i].tolist())[:, 0] != 0  # 逐个batch处理，根据offset_mapping判断该token是完整的词还是子词
    #         a = text_tampering_prob[i].tolist()
    #         word_tampering_prob = [pred for idx, pred in enumerate(text_tampering_prob[i].tolist()) if not is_subword[idx]]
    #         word_tampering_prob = torch.tensor(word_tampering_prob)
    #         for j, value in enumerate(gap_y[i]): # 逐个文本区域处理
    #             for id in range(value + 1):
    #                 left_top = ocr_points_left_top[i, j].item()
    #                 right_top = ocr_points_right_top[i, j].item()
    #                 index_left = left_top + (id * L)
    #                 index_right = right_top + (id * L)
    #                 position = np.arange(index_left, index_right + 1, dtype=np.longlong)
    #                 if len(all_positions) == 0:
    #                     all_positions.append(position)
    #                 else:
    #                     all_positions[0] = np.append(all_positions[0], position, axis=0)
    #         fusion_prob[i][all_positions[0]] = torch.mul(word_tampering_prob[i], visual_tampering_prob[i][all_positions[0]])
    #
    #     return fusion_prob

    def vtfeature_fusion(self, visual_features, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
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
        weighted_visual_features = visual_features.clone()
        for i in range(B):  # 逐个Batch处理
            num_nonzero = torch.sum(attention_mask[i] != 0).item() # 实际token长度，去除Pad
            is_subword = np.array(offset_mapping[i][:num_nonzero].tolist())[:, 0] != 0  # 逐个batch处理，根据offset_mapping判断该token是完整的词还是子词
            # a = text_tampering_prob[i].tolist()
            word_tampering_prob = [pred for idx, pred in enumerate(text_tampering_prob[i][:num_nonzero].tolist()) if not is_subword[idx]]
            word_tampering_prob = torch.tensor(word_tampering_prob).to(self.device)[1:-1] # 去除特殊词元<cls>和<sep>
            # print("nonzero:", num_nonzero)
            # print('bbox nonzero:', bbox[i][:num_nonzero].shape)
            word_bbox = [pred for idx, pred in enumerate(bbox[i][:num_nonzero].tolist()) if not is_subword[idx]]
            word_bbox = torch.tensor(word_bbox).to(self.device)[1:-1]

            index = torch.where((patch_original_bbox[i][:, None] == word_bbox).all(-1))[1].tolist() # 从整张Image中取出对应Patch的文本篡改概率
            patch_word_tampering_prob = word_tampering_prob[index]
            # print("patch_original_bbox", patch_original_bbox[i])
            # print("word_bbox", word_bbox)
            # print("index:", index)
            # print("patch_word_tampering_prob shape:", patch_word_tampering_prob.shape)
            # print("gay_y:", gap_y[i])
            # print("gay_y length:", len(gap_y[i]))

            for j, value in enumerate(gap_y[i]): # 逐个文本区域处理
                for id in range(value + 1):
                    left_top = ocr_points_left_top[i, j].item()
                    right_top = ocr_points_right_top[i, j].item()
                    if left_top == 0 and right_top == 0:
                        continue
                    index_left = left_top + (id * L)
                    index_right = right_top + (id * L)
                    position = np.arange(index_left, index_right + 1, dtype=np.longlong)
                    weighted_visual_features[i][position] = torch.mul(patch_word_tampering_prob[j].item(), visual_features[i][position])
        return weighted_visual_features


    def forward(self, input_ids=None, bbox=None, attention_mask=None, pixel_values=None, patch_norm_bbox=None,
                patch_original_bbox=None, visual_labels=None, offset_mapping=None, text_tampering_prob=None):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.model.prompt_embed_dim), device=self.device) # (1, 0 , 256)
        dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size) # (1, 256, 64, 64)

        visual_features = self.model.image_encoder(self.input) # (B, 256, 64, 64)
        B, C, H, W = visual_features.shape

        self.visual_features = visual_features.flatten(2).permute(0, 2, 1) # (B, 64*64, 256)
        # visual_tampering_logit = self.model.visual_classifier(self.visual_features)
        # self.visual_tampering_prob = F.softmax(visual_tampering_logit, dim=-1)
        # self.labels = self.generate_label(visual_labels)

        # output = self.model.DocAI(input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=pixel_values)
        # self.text_tampering_prob = F.softmax(output.logits, dim=-1)[:, :, 5].unsqueeze(-1) # (B, token_length, 1)

        num_patches = self.inp_size // self.patch_size
        self.fusion_features = self.vtfeature_fusion(self.visual_features, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
                                                    patch_original_bbox, offset_mapping, num_patches).permute(0, 2, 1).reshape(B, -1, H, W)

        # self.fusion_prob = self.vtfeature_fusion(self.visual_tampering_prob, self.text_tampering_prob, norm_bbox, offset_mapping, num_patches).permute(0, 2, 1).reshape(B, -1, H, W)
        # self.weighted_visual_features = self.visual_features * self.fusion_prob

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.fusion_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) # (B, 1, 256. 256) -> (B, 1, 1024, 1024)
        self.pred_mask = masks

    def infer(self, input, input_ids=None, bbox=None, attention_mask=None, pixel_values=None, patch_norm_bbox=None,
                patch_original_bbox=None, visual_labels=None, offset_mapping=None, text_tampering_prob=None):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.model.prompt_embed_dim), device=self.device) # (1, 0 , 256)
        dense_embeddings = self.model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size) # (1, 256, 64, 64)

        visual_features = self.model.image_encoder(input) # (B, 256, 64, 64)
        B, C, H, W = visual_features.shape

        self.visual_features = visual_features.flatten(2).permute(0, 2, 1) # (B, 64*64, 256)

        # output = self.model.DocAI(input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=pixel_values)
        # self.text_tampering_prob = F.softmax(output.logits, dim=-1)[:, :, 5].unsqueeze(-1) # (B, token_length, 1)

        num_patches = self.inp_size // self.patch_size
        self.fusion_features = self.vtfeature_fusion(self.visual_features, text_tampering_prob, bbox, attention_mask, patch_norm_bbox,
                                                     patch_original_bbox, offset_mapping, num_patches).permute(0, 2, 1).reshape(B, -1, H, W)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.fusion_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size) # (B, 1, 256. 256) -> (B, 1, 1024, 1024)
        return masks, self.fusion_features, visual_features

