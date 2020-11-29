# X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers



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

```
@inproceedings{Cho2020XLXMERT,
  title={X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers},
  author={Cho, Jaemin and Lu, Jiasen and Schwenk, Dustin and Hajishirzi, Hannaneh and Kembhavi, Aniruddha},
  booktitle={EMNLP},
  year={2020}
}
```