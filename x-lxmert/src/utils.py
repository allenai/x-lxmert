import torch
import collections
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_dict(_results, gpu=0, reduce_gpu=0, long=False):
    results = {}
    for k, v in _results.items():
        if type(v) == torch.Tensor:
            if long:
                results[k] = v.long().cuda(gpu)
            else:
                results[k] = v.float().cuda(gpu)
        else:
            if long:
                results[k] = torch.tensor(v).long().cuda(gpu)
            else:
                results[k] = torch.tensor(v).float().cuda(gpu)

    names = []
    values = []
    for k in sorted(results.keys()):
        names.append(k)
        values.append(results[k])
    values = torch.stack(values, dim=0)
    dist.reduce(values, dst=reduce_gpu)

    if gpu == 0:
        reduced_dict = {}
        for k, v in zip(names, values):
            reduced_dict[k] = v.item()
        return reduced_dict
    else:
        return None


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
    return new_state_dict


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def sum(self):
        return sum(self.vals)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def box_position(grid_size=5):
    n_grids = grid_size ** 2
    boxes = np.zeros(shape=(n_grids, 4), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            # pre-normalize (0 ~ 1)
            x0, x1 = j / grid_size, (j + 1) / grid_size
            y0, y1 = i / grid_size, (i + 1) / grid_size
            coordinate = (x0, y0, x1, y1)
            boxes[i * grid_size + j] = coordinate
    return boxes
