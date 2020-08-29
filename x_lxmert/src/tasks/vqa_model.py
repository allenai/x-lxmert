# coding=utf-8
# Copyleft 2019 project LXRT.

from lxrt.modeling import GeLU, BertLayerNorm, LXRTModel, BertPreTrainedModel, BertVisualAnswerHead

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# from param import args
# from lxrt.entry import LXRTEncoder
# from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(BertPreTrainedModel):
    def __init__(self, config, args, num_answers):
        super().__init__(config)

        self.config = config
        self.args = args
        # self.mode = mode

        # Build LXRT encoder
        # from lxrt.entry import LXRTEncoder2
        # self.lxrt_encoder = LXRTEncoder2(
        #     args,
        #     max_text_length=MAX_VQA_LENGTH,
        #     mode=mode
        # )
        # hid_dim = self.lxrt_encoder.dim

        self.bert = LXRTModel(config, args)

        hid_dim = self.config.hidden_size

        self.answer_head = BertVisualAnswerHead(config, num_answers)

        # VQA Answer heads
        # self.logit_fc = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim * 2),
        #     GeLU(),
        #     BertLayerNorm(hid_dim * 2, eps=1e-12),
        #     nn.Linear(hid_dim * 2, num_answers)
        # )
        # self.logit_fc.apply(self.init_bert_weights)

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
                if args.im_ratio == 'original':
                    centroid_path = centroid_dir.joinpath(
                        f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.npy')
                else:
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
        code, grid = visn_feats
        word_id, token_type_ids, word_attention_mask = sent_feats

        (lang_output, visn_output), pooled_output = self.bert(
            word_id, token_type_ids, word_attention_mask,
            visual_feats=(code, grid),
            skip_vis_fc=False,
            get_lang_cls_feat=False,
            visual_AR=False)

        # logit = self.logit_fc(pooled_output)
        logit = self.answer_head(pooled_output)

        return logit

    def forward_backup_2(self, visn_feats, sent_feats):
        """
        visn_feats
            feat: B, n_grids, feat_dim
            pos:  B, n_grids, 4
        sent_feats
            input_ids: B, max_text_length
            token_type_ids: B, max_text_length
            word_attention_mask: B, max_text_length
        """
        feat, pos = visn_feats
        B, obj_num, feat_size = feat.size()
        assert obj_num == self.grid_size**2 and feat_size == self.args.feat_dim

        if self.mode == 'x':
            # Use features: pooled_output [CLS]
            x = self.lxrt_encoder(sent_feats, visn_feats)  # [B, hid_dim]
            logit = self.logit_fc(x)  # [B, num_answers]
            return logit

    def forward_backup_1(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit
