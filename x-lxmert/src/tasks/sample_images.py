import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
to_pil_image = transforms.ToPILImage()

from transformers import LxmertTokenizer

from utils import box_position, load_state_dict
from param import parse_args
from src.tasks.imggen_model import ImggenModel

tokenizer = LxmertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

args = parse_args()
args.n_grids = args.grid_size**2

args.gpu = torch.cuda.current_device()
# print(args)

def clean_text(sent):
    sent = sent.replace("\ufffd\ufffd", " ")
    sent = sent.replace("\n", ' ')
    sent = sent.replace(" .", '.')
    sent = " ".join(sent.split())
    return sent

def get_batch_entry(sentences, seq_length=20):
    batch_entry = {}

    B = len(sentences)
    max_L = 0

    input_ids = torch.zeros(B, seq_length).long()

    for i, sent in enumerate(sentences):
        tokens = tokenizer.tokenize(sent.strip())
        # Account for [CLS] and [SEP] with "- 2"
        if seq_length is not None:
            if len(tokens) > seq_length - 2:
                tokens = tokens[:(seq_length - 2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        L = len(token_ids)
        max_L = max(L, max_L)
        input_ids[i, :L] += torch.LongTensor(token_ids)

    input_ids = input_ids[:, :max_L]

    # language Inputs
    batch_entry['word_id'] = input_ids
    batch_entry['sent'] = sentences

    batch_entry['boxes'] = torch.from_numpy(
        box_position(args.grid_size)).unsqueeze(0).expand(B, -1, -1)
    batch_entry['vis_mask'] = torch.zeros(B, args.grid_size**2)
    batch_entry['cluster_id'] = torch.zeros(B, args.grid_size**2).long()

    return batch_entry

if __name__ == '__main__':

    from time import time

    start = time()

    model = ImggenModel.from_pretrained("bert-base-uncased")

    ckpt_path = Path(__file__).resolve().parent.parent.parent.joinpath('snap/pretrained/x_lxmert/Epoch20_LXRT.pth')
    state_dict = load_state_dict(ckpt_path, 'cpu')

    results = model.load_state_dict(state_dict, strict=False)
    print(results)


    img_save_dir = Path('./img_samples').resolve()
    # intermediate_img_save_dir = Path('./img_samples/intermediate').resolve()
    # intermediate_img_save_dir.mkdir(exist_ok=True, parents=True)

    model.img_log_dir = img_save_dir

    print(f'Loaded model | {time() - start:.2f}s')

    with open('./example_sentences.txt') as f:
        captions = f.readlines()

    print('Loaded captions')
    print(captions)

    entry = get_batch_entry(captions)

    print(f'prepared batch entry | {time() - start:.2f}s')

    # get image
    generated_imgs = [0]*len(captions)
    generated_imgs = model.sample(entry, -1, -1,
                                    n_steps=args.sample_steps,
                                    out_intermediate=True,
                                    custom_img=True,
                                    seed=None,
                                    save_sent=False,
                                    return_imgs=False)

    print(f'Inference done | {time() - start:.2f}s')


    #
    # for caption, img_tensor in zip(captions, generated_imgs):
    #
    #     img = to_pil_image(img_tensor).convert("RGB")
    #     fname = f'{caption}.png'
    #     fpath = img_log_dir.joinpath(fname)
    #     img.save(fpath)


    print(f'Images saved at {img_save_dir} | {time() - start:.2f}s')
