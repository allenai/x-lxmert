import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from transformers import LxmertTokenizer

from utils import box_position, load_state_dict
from param import parse_args
from tasks.imggen_model import ImggenModel

from layers import Generator

def clean_text(sent):
    sent = sent.replace("\ufffd\ufffd", " ")
    sent = sent.replace("\n", ' ')
    sent = sent.replace(" .", '.')
    sent = " ".join(sent.split())
    return sent

if __name__ == '__main__':
    args = parse_args()
    args.n_grids = args.grid_size**2
    # args.gpu = torch.cuda.current_device()

    from time import time

    start = time()

    # 1) Load X-LXMERT
    model = ImggenModel.from_pretrained("bert-base-uncased")

    ckpt_path = Path(__file__).resolve().parents[2].joinpath('snap/pretrained/x_lxmert/Epoch20_LXRT.pth')
    state_dict = load_state_dict(ckpt_path, 'cpu')

    results = model.load_state_dict(state_dict, strict=False)
    print(results)
    print(f'Loaded X-LXMERT | {time() - start:.2f}s')

    # 2) Load Visual embedding
    clustering_dir = args.datasets_dir.joinpath('clustering')
    centroid_path = clustering_dir.joinpath(f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.npy')
    centroids = np.load(centroid_path)
    model.set_visual_embedding(centroids)

    # 2) Load Generator
    code_dim = 256
    SN = True
    base_dim = 32
    # img_target_size = 256

    G = Generator(
        base_dim=base_dim,
        emb_dim=args.feat_dim,
        norm_type=args.norm_type,
        target_size=args.resize_target_size,
        init_H=args.grid_size,
        init_W=args.grid_size,
        SN=SN,
        codebook_dim=code_dim,
    )

    model.set_image_generator(G)

    model.cuda()
    model.eval()

    with open('./example_sentences.txt') as f:
        captions = f.readlines()
    captions = [clean_text(sent) for sent in captions]

    print('Loaded captions')
    print(captions)

    print(f'prepared batch entry | {time() - start:.2f}s')

    # 3) Sample Images
    generated_imgs = model.sample_image_NAR(captions,
                                            max_text_length=args.max_text_length
                                            n_steps=args.sample_steps,
                                            return_intermediate=False)

    print(f'Inference done | {time() - start:.2f}s')

    img_save_dir = Path('./img_samples').resolve()
    # intermediate_img_save_dir = Path('./img_samples/intermediate').resolve()
    # intermediate_img_save_dir.mkdir(exist_ok=True, parents=True)
    img_save_dir.mkdir(exist_ok=True)

    to_pil_image = transforms.ToPILImage()

    for caption, img_tensor in zip(captions, generated_imgs):
        img = to_pil_image(img_tensor).convert("RGB")
        fname = f'{caption}.png'
        fpath = img_log_dir.joinpath(fname)
        img.save(fpath)

    print(f'Images saved at {img_save_dir} | {time() - start:.2f}s')
