# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import copy
from pathlib import Path
import torch

from utils import load_state_dict

class AnswerTable:
    ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, args, dsets=None):
        self.args = args
        self.datasets_dir = Path(self.args.datasets_dir)
        # self.all_ans = json.load(open("data/lxmert/all_ans.json"))
        all_ans_path = self.datasets_dir.joinpath(
            'data/lxmert/all_ans.json')
        self.all_ans = json.load(open(all_ans_path))
        if dsets is not None:
            dsets = set(dsets)
            # If the answer is used in the dsets
            self.anss = [ans['ans'] for ans in self.all_ans if
                         len(set(ans['dsets']) & dsets) > 0]
        else:
            self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id,
                            ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)


def load_lxmert_qa(args, path, model, label2ans, verbose=False, loc='cpu'):
    """
    Load model weights from LXMERT pre-training.
    The answers in the fine-tuned QA task (indicated by label2ans)
        would also be properly initialized with LXMERT pre-trained
        QA heads.

    :param path: Path to LXMERT snapshot.
    :param model: LXRT model instance.
    :param label2ans: The label2ans dict of fine-tuned QA datasets, like
        {0: 'cat', 1: 'dog', ...}
    :return:
    """
    if verbose:
        print("Load QA pre-trained LXMERT from %s " % path)

    loaded_state_dict = load_state_dict(path, loc)
    model_state_dict = model.state_dict()

    # Do surgery on answer state dict
    ans_weight = loaded_state_dict['answer_head.logit_fc.3.weight']
    ans_bias = loaded_state_dict['answer_head.logit_fc.3.bias']

    new_answer_weight = copy.deepcopy(model_state_dict['answer_head.logit_fc.3.weight'])
    new_answer_bias = copy.deepcopy(model_state_dict['answer_head.logit_fc.3.bias'])
    answer_table = AnswerTable(args)
    loaded = 0
    unload = 0
    if type(label2ans) is list:
        label2ans = {label: ans for label, ans in enumerate(label2ans)}
    for label, ans in label2ans.items():
        new_ans = answer_table.convert_ans(ans)
        if answer_table.used(new_ans):
            ans_id_9500 = answer_table.ans2id(new_ans)
            new_answer_weight[label] = ans_weight[ans_id_9500]
            new_answer_bias[label] = ans_bias[ans_id_9500]
            loaded += 1
        else:
            new_answer_weight[label] = 0.
            new_answer_bias[label] = 0.
            unload += 1
    if verbose:
        print("Loaded %d answers from LXRTQA pre-training and %d not" % (loaded, unload))
        print()
    loaded_state_dict['answer_head.logit_fc.3.weight'] = new_answer_weight
    loaded_state_dict['answer_head.logit_fc.3.bias'] = new_answer_bias

    result = model.load_state_dict(loaded_state_dict, strict=False)
    if verbose:
        print(result)
