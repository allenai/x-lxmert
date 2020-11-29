from transformers.modeling_lxmert import LxmertVisualAnswerHead, LxmertModel, LxmertPreTrainedModel

import torch
import torch.nn as nn

class VQAModel(LxmertPreTrainedModel):
    def __init__(self, config, num_answers, num_clusters=10000):
        super().__init__(config)

        self.config = config
        self.num_answers = num_answers
        self.config.num_clusters = num_clusters
        self.config.clustering = num_clusters > 0

        self.bert = LxmertModel(config)

        # VQA Answer head
        self.answer_head = LxmertVisualAnswerHead(config, self.config.num_answers)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def set_visual_embedding(self, centroids):
        self.vis_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(centroids),
            freeze=True
        )

    def forward(self,
                input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                visual_attention_mask=None,

                cluster_ids=None,
                # vis_mask=None,

                token_type_ids=None,
                inputs_embeds=None,

                return_dict=True,
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        out_dict = {}

        if self.config.clustering:
            visual_feats = self.vis_emb(cluster_ids)

        B, V_L, _ = visual_feats.size()

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

        logit = self.answer_head(pooled_output)

        return logit

        # out_dict['logit'] = logit
        # out_dict['pred'] = logit.argmax(dim=-1).detach().flatten().cpu().numpy()

        # return out_dict

    # def train_step(self, batch):


    # def test_step(self, batch):


