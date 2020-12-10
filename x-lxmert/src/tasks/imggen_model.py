import random

import numpy as np
import torch
import torch.nn as nn

from transformers.modeling_lxmert import LxmertModel, LxmertPreTrainedModel, LxmertVisualObjHead
from utils import box_position
from transformers import LxmertTokenizer

class ImggenModel(LxmertPreTrainedModel):
    def __init__(self, config, args, num_clusters=10000):
        super().__init__(config)

        self.config = config
        self.args = args
        self.config.num_clusters = num_clusters
        self.config.clustering = num_clusters > 0

        self.bert = LxmertModel(config)

        self.obj_predict_head = LxmertVisualObjHead(config)

        self.mask_feat = nn.Parameter(torch.zeros(config.visual_feat_dim))
        self.vis_emb = None

        self.tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')

    def set_visual_embedding(self, centroids):
        # import numpy as np
        if isinstance(centroids, np.ndarray):
            centroids = torch.from_numpy(centroids)
        elif isinstance(centroids, torch.Tensor):
            pass
        self.vis_emb = nn.Embedding.from_pretrained(
            centroids,
            freeze=True
        )
        self.obj_predict_head.out_cluster.weight = self.vis_emb.weight

    def set_image_generator(self, generator):
        self.G = generator

    def denorm(self, x):
        """(-1, 1) => (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def sample_image_AR(self,
                        sentences,
                        max_text_length=20,
                        position_random=False,
                        position_TLBR=False,
                        position_confidence=True,
                        n_steps=None,
                        seed=None,
                        return_intermediate=False,
                        ):
        B = len(sentences)

        input_ids = self.tokenizer(
            sentences, max_length=max_text_length, truncation=True, return_tensors='pt').input_ids
        input_ids = input_ids.cuda()

        grid_size = 8
        n_grids = grid_size**2
        code_dim = 2048

        if n_steps is None:
            n_steps = grid_size ** 2

        visual_pos = torch.from_numpy(box_position(grid_size)).unsqueeze(0).expand(B, -1, -1).cuda()

        intermediate_imgs = []

        if position_random:
            positions = list(range(grid_size ** 2))
            if seed is not None:
                random.Random(seed).shuffle(positions)
            else:
                random.shuffle(positions)
            if n_steps > grid_size ** 2:
                additional_positions = list(range(n_steps - grid_size ** 2))
                if seed is not None:
                    random.Random(seed).shuffle(additional_positions)
                else:
                    random.shuffle(additional_positions)
                positions = additional_positions + positions

        if position_confidence:
            visited_positions = torch.zeros(B, grid_size**2).cuda()

        for i in range(n_steps):

            if i == 0:
                vis_mask = torch.ones(B, grid_size**2).long().cuda()
                code = torch.zeros(B, grid_size**2, code_dim).cuda()

            if position_random:
                current_pos_i = positions.pop()
                current_pos_i = current_pos_i % grid_size**2
                # for more than NxN iteration
                vis_mask[:, current_pos_i] = 1
            elif position_TLBR:
                current_pos_i = i

            # Masking visual feats
            code = torch.where(vis_mask.view(B, n_grids, 1).bool(),
                                self.mask_feat.view(1, 1, -1).to(dtype=code.dtype),
                                code)

            # LXMERT Forward Propagation
            lxmert_output = self.bert(
                input_ids=input_ids,
                visual_feats=code,
                visual_pos=visual_pos,
                attention_mask=input_ids > 0,
                return_dict=True,
            )
            visual_output = lxmert_output[1]
            pred_code_logit = self.obj_predict_head(visual_output, out_keys=['obj'])['obj']

            # [B, n_grids, n_codes]
            pred_code_prob = torch.softmax(pred_code_logit, dim=2)

            # [B, n_grids]
            pred_prob, pred_code_id = pred_code_prob.max(dim=2)

            # [B, n_grids, code_dim]
            pred_code = self.vis_emb(pred_code_id)

            # Update masked codes
            if position_TLBR or position_random:
                update_mask = torch.zeros(B, grid_size**2).bool().cuda()
                update_mask[:, current_pos_i] = 1
                vis_mask[:, current_pos_i] = 0

            elif position_confidence:
                _pred_prob = pred_prob.masked_fill(
                    visited_positions.bool(), -10000)

                top_prob, top_arg = _pred_prob.topk(1, dim=1, largest=True)
                update_mask = torch.zeros(B, grid_size**2).long().cuda()
                update_mask.scatter_(1, top_arg, 1)
                vis_mask.scatter_(1, top_arg, 0)

                visited_positions.scatter_(1, top_arg, 1)

            code = torch.where(update_mask.view(B, grid_size**2, 1).bool(),
                               pred_code,
                               code)

            if return_intermediate:
                fake_img = self.G(code.permute(0, 2, 1).view(B, code_dim, grid_size, grid_size))
                fake_img = self.denorm(fake_img).cpu()
                intermediate_imgs.append(fake_img)

        if return_intermediate:
            return intermediate_imgs

        # 4) Generate image with Pre-trained Decoder
        fake_img = self.G(code.permute(0, 2, 1).view(B, code_dim, grid_size, grid_size))
        fake_img = self.denorm(fake_img).cpu()

        return fake_img

    def sample_image_NAR(self,
                        sentences,
                        max_text_length=20,
                        # position_random=False,
                        # position_TLBR=False,
                        # position_confidence=True,
                        n_steps=None,
                        # seed=None,
                        return_intermediate=False,
                        ):
        B = len(sentences)

        input_ids = self.tokenizer(
            sentences, max_length=max_text_length, truncation=True, return_tensors='pt').input_ids
        input_ids = input_ids.cuda()

        grid_size = 8
        n_grids = grid_size**2
        code_dim = 2048

        if n_steps is None:
            n_steps = grid_size ** 2

        visual_pos = torch.from_numpy(box_position(grid_size)).unsqueeze(0).expand(B, -1, -1).cuda()

        intermediate_imgs = []

        for i in range(n_steps):
            # Linear decay for mask updates (Mask-Predict)
            ratio = (n_steps - i) / n_steps
            n_mask = int(ratio * grid_size**2)

            if i == 0:
                vis_mask = torch.ones(B, grid_size**2).long().cuda()
                code = torch.zeros(B, grid_size**2, code_dim).cuda()
            else:
                # [B, n_grids]
                lowest_prob, lowest_arg = pred_prob.topk(
                    n_mask, dim=1, largest=False)
                vis_mask = torch.zeros(B, grid_size**2).long().cuda()
                vis_mask.scatter_(1, lowest_arg, 1)

            # Masking visual feats
            code = torch.where(vis_mask.view(B, n_grids, 1).bool(),
                               self.mask_feat.view(
                                   1, 1, -1).to(dtype=code.dtype),
                               code)

            # LXMERT Forward Propagation
            lxmert_output = self.bert(
                input_ids=input_ids,
                visual_feats=code,
                visual_pos=visual_pos,
                attention_mask=input_ids > 0,
                return_dict=True,
            )
            visual_output = lxmert_output[1]
            pred_code_logit = self.obj_predict_head(visual_output, out_keys=['obj'])['obj']

            # [B, n_grids, n_codes]
            pred_code_prob = torch.softmax(pred_code_logit, dim=2)

            # [B, n_grids]
            pred_prob, pred_code_id = pred_code_prob.max(dim=2)

            # [B, n_grids, code_dim]
            pred_code = self.vis_emb(pred_code_id)

            # Update masked codes
            code = torch.where(vis_mask.view(B, grid_size**2, 1).bool(),
                                pred_code,
                                code)

            if return_intermediate:
                fake_img = self.G(code.permute(0, 2, 1).view(B, code_dim, grid_size, grid_size))
                fake_img = self.denorm(fake_img).cpu()
                intermediate_imgs.append(fake_img)

        if return_intermediate:
            return intermediate_imgs

        # 4) Generate image with Pre-trained Decoder
        fake_img = self.G(code.permute(0, 2, 1).view(B, code_dim, grid_size, grid_size))
        fake_img = self.denorm(fake_img).cpu()

        return fake_img
