# coding=utf-8
# Copyleft 2019 project LXRT.

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from lxrt.modeling import GeLU, BertLayerNorm, LXRTModel, BertPreTrainedModel
from lxrt.entry import LXRTEncoder2
# from param import args


class NLVR2Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.config = config
        self.args = args

        self.bert = LXRTModel(config, args)

        hid_dim = self.config.hidden_size
        self.hid_dim = hid_dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 2)
        )
        self.logit_fc.apply(self.init_bert_weights)

        self.grid_size = args.grid_size

        if self.args.clustering:
            if self.args.vis_vocab_from_scratch:
                Emb = nn.Embedding(args.n_codebook, self.config.hidden_size)
            else:
                centroid_dir = Path(
                    '/home/jaeminc/Dropbox/Projects/AI2/clustering/').resolve()
                if args.v4:
                    centroid_dir = centroid_dir.joinpath('v4')
                imsize = args.resize_input_size
                centroid_path = centroid_dir.joinpath(
                    f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.npy')
                centroids = np.load(centroid_path)

                Emb = nn.Embedding.from_pretrained(
                    torch.from_numpy(centroids),
                    freeze=not self.args.vis_vocab_tune,
                    # scale_grad_by_freq=True
                )
            self.vis_emb = Emb


    def forward(self, visn_feats, sent_feats):
        """
        visn_feats
            feat: B, 2, n_grids, feat_dim
            pos:  B, 2, n_grids, 4
        sent_feats
            input_ids: B * 2, max_text_length
            token_type_ids: B * 2, max_text_length
            word_attention_mask: B * 2, max_text_length
        """
        # Pairing images and sentences:
        # The input of NLVR2 is two images and one sentence. In batch level, they are saved as
        #   [ [img0_0, img0_1], [img1_0, img1_1], ...] and [sent0, sent1, ...]
        # Here, we flat them to
        #   feat/pos = [ img0_0, img0_1, img1_0, img1_1, ...]
        #   sent     = [ sent0,  sent0,  sent1,  sent1,  ...]

        vis_feat, pos = visn_feats
        B, img_num, obj_num, feat_size = vis_feat.size()
        # assert img_num == 2 and obj_num == 36 and feat_size == 2048
        assert img_num == 2
        if self.args.grid_model:
            assert obj_num == self.grid_size**2
        else:
            assert obj_num == self.args.n_boxes
        assert feat_size == self.args.feat_dim

        # Concatenate two images features
        vis_feat = vis_feat.view(B * 2, obj_num, feat_size)
        pos = pos.view(B * 2, obj_num, 4)

        word_id, token_type_ids, word_attention_mask = sent_feats

        (lang_output, visn_output), pooled_output = self.bert(
            word_id, token_type_ids, word_attention_mask,
            visual_feats=(vis_feat, pos),
            skip_vis_fc=False,
            get_lang_cls_feat=False,
            visual_AR=False)

        # Use features: pooled_output [CLS]
        x = pooled_output.view(B, 2 * self.hid_dim)
        logit = self.logit_fc(x)  # [B, 2]

        return logit

        # if self.mode == 'x':
        #     # Use features: pooled_output [CLS]
        #     x = self.lxrt_encoder(sent_feats, visn_feats, get_lang_cls_feat=False)  # [B * 2, hid_dim]
        #     x = x.view(B, 2 * self.hid_dim)
        #     logit = self.logit_fc(x)  # [B, 2]
        #     return logit

        # else:
        #     feat_seq, pooled_output = self.lxrt_encoder(sent_feats, visn_feats)  # [B * 2, hid_dim]
        #     # feat_seq = (lang_output, visn_output) # [B * 2, n_grid + max_len, hid_dim]
        #     feat_1, feat_2 = feat_seq
        #
        #     return logit
