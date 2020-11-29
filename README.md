# X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers ([EMNLP 2020](https://2020.emnlp.org/))

* Authors: [Jaemin Cho](https://j-min.io), [Jiasen Lu](https://www.cc.gatech.edu/~jlu347/), [Dustin Schwenk](https://www.semanticscholar.org/author/D.-Schwenk/34846449), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), and [Ani Kembhavi](https://anikem.github.io/).
* [Paper](https://arxiv.org/abs/2009.11278)
* [Blog](https://prior.allenai.org/projects/x-lxmert)
* [Demo](https://vision-explorer.allenai.org/text_to_image_generation)
* [Slideslive Presentation](https://slideslive.com/38938675/xlxmert-paint-caption-and-answer-questions-with-multimodal-transformers)

# Summary
Recent multi-modal transformers have achieved tate of the art performance on a variety of multimodal discriminative tasks like visual question answering and generative tasks like image captioning. This begs an interesting question: Can these models go the other way and generate images from pieces of text? Our analysis of a popular representative from this model family - LXMERT - finds that it is unable to generate rich and semantically meaningful imagery with its current training setup. We introduce X-LXMERT, an extension to LXMERT with training refinements. X-LXMERT's image generation capabilities rival state of the art generative models while its question answering and captioning abilities remains comparable to LXMERT.

![sample images](https://prior.allenai.org/assets/project-content/x-lxmert/generation_process.gif)

# Install

```
pip install -r ./requirements.txt
```

# Feature extraction

Please check [./feature_extraction/README.md](./feature_extraction/README.md) for more details.

```bash
cd ./feature_extraction

python coco_extract_grid_feature.py
python VG_extract_grid_feature.py
python GQA_extract_grid_feature.py
python nlvr2_extract_grid_feature.py

python run_kmeans.py
```

# Pretraining

## Pretrain on LXMERT Pretraining data
```bash
cd ./x-lxmert/
bash scripts/pretrain.bash
```

## or download pretrained checkpoint
```
wget -O x-lxmert/snap/pretrained/x_lxmert/Epoch20_LXRT.pth https://storage.googleapis.com/x_lxmert/Epoch20_LXRT.pth
```

# Finetuning

## VQA
```bash
cd ./x-lxmert/
bash scripts/finetune_vqa.bash
```

## GQA
```bash
cd ./x-lxmert/
bash scripts/finetune_gqa.bash
```

## NLVR2
```bash
cd ./x-lxmert/
bash scripts/finetune_nlvr2.bash
```

# Image generation

## Train image generator on MS COCO
```bash
cd ./x-lxmert/
bash scripts/generate_image.bash
```

## or download pretrained checkpoints
```
wget -O x-lxmert/snap/pretrained/x_lxmert/Epoch20_LXRT.pth https://storage.googleapis.com/x_lxmert/Epoch20_LXRT.pth
```


# Reference

```BibTex
@inproceedings{Cho2020XLXMERT,
  title={X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers},
  author={Cho, Jaemin and Lu, Jiasen and Schwenk, Dustin and Hajishirzi, Hannaneh and Kembhavi, Aniruddha},
  booktitle={EMNLP},
  year={2020}
}
```