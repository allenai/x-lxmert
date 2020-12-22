import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from transformers.models.lxmert.modeling_lxmert import LxmertPredictionHeadTransform, LxmertPreTrainingHeads, LxmertVisualAnswerHead, LxmertModel, LxmertPreTrainedModel


class LxmertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,), "num": config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,), "num": config.num_attr_labels}
        if config.visual_obj_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
            }
        self.visual_losses = visual_losses

        self.linear_feat = nn.Linear(config.hidden_size, config.visual_feat_dim)
        if config.num_clusters > 0:
            self.cluster_out = True
            self.out_cluster = nn.Linear(config.visual_feat_dim, config.num_clusters)
        else:
            self.cluster_out = False
            if 'obj' in self.visual_losses:
                self.out_obj = nn.Linear(config.visual_feat_dim, config.num_object_labels)
            if 'attr' in self.visual_losses:
                self.out_attr = nn.Linear(config.visual_feat_dim, config.num_attr_labels)

    def forward(self, hidden_states, out_keys=[]):
        hidden_states = self.transform(hidden_states)

        output = {}
        feat = self.linear_feat(hidden_states)
        if 'feat' in self.visual_losses or 'feat' in out_keys:
            output['feat'] = feat
        if 'obj' in self.visual_losses or 'obj' in out_keys:
            if self.cluster_out:
                output['obj'] = self.out_cluster(feat)
            else:
                output['obj'] = self.out_obj(feat)
        if 'attr' in self.visual_losses or 'attr' in out_keys:
            output['attr'] = self.out_attr(feat)

        return output


class XLxmertForPretraining(LxmertPreTrainedModel):
    def __init__(self, config, num_clusters=10000):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        # self.visual_loss_normalizer = config.visual_loss_normalizer

        self.config.num_clusters = num_clusters
        self.config.clustering = num_clusters > 0

        # Use of pre-training tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        self.verbose = True
        if torch.cuda.is_available() and torch.cuda.current_device() != 0:
            self.verbose = False

        # Lxmert backbone
        if self.verbose:
            print('Bulding LXMERT backbone')
        self.bert = LxmertModel(config)

        if self.verbose:
            print('Bulding Pretrain Heads')
        # Pre-training heads
        if self.task_mask_lm or self.task_matched:
            self.cls = LxmertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        self.mask_feat = nn.Parameter(torch.zeros(config.visual_feat_dim))
        self.vis_emb = None

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "visual_ce": CrossEntropyLoss(reduction="none"),
            "ce": CrossEntropyLoss(),
        }

        self.cross_entropy = CrossEntropyLoss()
        self.huber_loss = SmoothL1Loss(reduction='none')

        if self.verbose:
            print('Weight initialization')

        # Weight initialization
        self.init_weights()

        visual_losses = {}
        if config.visual_obj_loss:
            if self.config.clustering:
                visual_losses["obj"] = {
                    "shape": (-1,),
                    "num": config.num_clusters,
                    "loss": "visual_ce",
                }
            else:
                visual_losses["obj"] = {
                    "shape": (-1,),
                    "num": config.num_object_labels,
                    "loss": "visual_ce",
                }
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),
                "num": config.num_attr_labels,
                "loss": "visual_ce",
            }
        if config.visual_obj_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        self.visual_losses = visual_losses


    def set_visual_embedding(self, centroids):
        import numpy as np
        if isinstance(centroids, np.ndarray):
            centroids = torch.from_numpy(centroids)
        elif isinstance(centroids, torch.Tensor):
            pass
        self.vis_emb = nn.Embedding.from_pretrained(
            centroids,
            freeze=True
        )
        if self.task_obj_predict:
            self.obj_predict_head.out_cluster.weight = self.vis_emb.weight


    def forward(self,
                input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                visual_attention_mask=None,

                cluster_ids=None,
                vis_mask=None,

                token_type_ids=None,
                inputs_embeds=None,
                # labels=None,
                # obj_labels=None,
                # matched_label=None,
                # ans=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,

                label_dict=None,
                task=['word_mask', 'vis_mask', 'matched', 'qa'],
                **kwargs
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        out_dict = {}

        if self.config.clustering:
            visual_feats = self.vis_emb(cluster_ids)

        B, V_L, _ = visual_feats.size()

        if task == 'vis_mask':
            visual_feats = torch.where(vis_mask.view(B, V_L, 1).bool(),
                                    self.mask_feat.view(1, 1, -1).to(dtype=visual_feats.dtype),
                                    visual_feats)

        lxmert_output = self.bert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        lang_output, visual_output, pooled_output = (
            lxmert_output[0],
            lxmert_output[1],
            lxmert_output[2],
        )

        total_loss = torch.tensor(0.0, device=device)

        if task in ['word_mask', 'matched', 'qa']:
            lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)

        if task == 'word_mask':
            word_labels = label_dict['word_labels']
            lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size),
                word_labels.view(-1)
            )
            total_loss += lm_loss
            out_dict['lm_loss'] = lm_loss.detach()

        if task == 'matched':
            matched_labels = label_dict['matched_labels']
            matched_loss = self.loss_fcts["ce"](
                cross_relationship_score.view(-1, 2),
                matched_labels.view(-1)
            )
            total_loss += matched_loss
            out_dict['matched_loss'] = matched_loss.detach()

        if task == 'vis_mask':
            visual_prediction_scores_dict = self.obj_predict_head(visual_output, self.visual_losses)

            total_vis_loss = torch.tensor(0.0, device=device)

            for key in self.visual_losses:

                if key == 'obj':
                    obj_logit = visual_prediction_scores_dict['obj']
                    if self.config.clustering:
                        n_objs = self.config.n_centroids
                    else:
                        n_objs = self.config.num_object_labels
                    assert obj_logit.size() == (B, V_L, n_objs), (obj_logit.size(), (B, V_L, n_objs))

                    obj_label = label_dict['obj_labels']
                    obj_loss = self.cross_entropy(
                        obj_logit.view(B*V_L, n_objs),
                        obj_label.flatten()
                    )
                    total_vis_loss += obj_loss
                    out_dict['obj_loss'] = obj_loss.detach()

                elif key == 'attr':
                    attr_logit = visual_prediction_scores_dict['attr']
                    assert attr_logit.size() == (B, V_L, self.config.num_attr_labels), attr_logit.size()

                    attr_label = label_dict['attr_labels']
                    attr_loss = self.cross_entropy(
                        attr_logit.view(B*V_L, self.config.num_attr_labels),
                        attr_label.flatten()
                    )

                    total_vis_loss += attr_loss
                    out_dict['attr_loss'] = attr_loss.detach()

                elif key == 'feat':
                    pred_feat = visual_prediction_scores_dict['feat']
                    feat_label = label_dict['feat_labels']
                    feat_loss = self.huber_loss(
                        pred_feat.view(B, V_L, self.config.visual_feat_dim),
                        feat_label.view(B, V_L, self.config.visual_feat_dim)
                    )  # [B, V_L, feat_dim]
                    feat_loss = feat_loss.mean(dim=2)  # [B, n_grids]
                    feat_loss = (feat_loss * vis_mask).sum(dim=1)  # [B]
                    n_mask = vis_mask.sum(dim=1).clamp(min=1)  # [B]
                    feat_loss = feat_loss / n_mask  # [B]
                    feat_loss = feat_loss.mean()

                    total_vis_loss += feat_loss
                    out_dict['feat_loss'] = feat_loss.detach()

            total_loss += total_vis_loss
            out_dict['vis_loss'] = total_vis_loss.detach()

        if self.task_qa:
            ans = label_dict['qa_labels']
            answer_score = self.answer_head(pooled_output)
            qa_loss = self.cross_entropy(
                answer_score.view(-1, self.num_qa_labels),
                ans.view(-1)
            )

            score, qa_pred = answer_score.max(1)

            total_loss += qa_loss
            out_dict['qa_loss'] = qa_loss.detach()
            out_dict['qa_pred'] = qa_pred

        out_dict['total_loss'] = total_loss

        return out_dict

if __name__ == '__main__':
    model = XLxmertForPretraining.from_pretrained("bert-base-uncased")

    from pathlib import Path
    from utils import box_position, load_state_dict

    ckpt_path = Path(__file__).resolve().parents[2].joinpath('snap/pretrained/x_lxmert/Epoch20_LXRT.pth')
    state_dict = load_state_dict(ckpt_path, 'cpu')

    results = model.load_state_dict(state_dict, strict=False)
    print(results)
