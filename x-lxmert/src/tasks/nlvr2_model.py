from transformers.models.lxmert.modeling_lxmert import LxmertVisualAnswerHead, LxmertModel, LxmertPreTrainedModel

import torch
import torch.nn as nn


class NLVR2Model(LxmertPreTrainedModel):
    def __init__(self, config, num_answers=2, num_clusters=-1):
        super().__init__(config)

        self.config = config
        self.num_answers = num_answers
        self.config.num_clusters = num_clusters
        self.config.clustering = num_clusters > 0

        self.bert = LxmertModel(config)

        self.logit_fc = LxmertVisualAnswerHead(config, num_answers)
        self._init_weights(self.logit_fc)

    def forward(self,
                input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                visual_attention_mask=None,

                # cluster_ids=None,
                # vis_mask=None,

                token_type_ids=None,
                inputs_embeds=None,

                return_dict=True,
                ):
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        out_dict = {}

        # if self.config.clustering:
        #     visual_feats = self.vis_emb(cluster_ids)

        B, n_images, V_L, feat_dim = visual_feats.size()
        assert n_images == 2

        # Concatenate two images features
        visual_feats = visual_feats.view(B * 2, V_L, feat_dim)
        visual_pos = visual_pos.view(B * 2, V_L, -1)

        lxmert_output = self.bert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            # output_hidden_states=output_hidden_states,
            # output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # Use features: pooled_output [CLS]
        pooled_output = lxmert_output[2]

        pooled_output = pooled_output.view(B, 2 * self.config.hidden_size)

        logit = self.answer_head(pooled_output)

        # return logit

        out_dict['logit'] = logit
        # out_dict['pred'] = logit.argmax(dim=-1).detach().flatten().cpu().numpy()

        return out_dict
