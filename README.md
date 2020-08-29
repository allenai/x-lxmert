

# Install

```
pip -r requirements.txt

or

conda install pytorch torchvision -c pytorch
conda install pyyaml boto3 h5py jsonlines requests boto3
conda install jupyterlab nb_conda numpy pandas matplotlib
conda install tqdm tensorboard pillow

pip install pycocotools
pip install pytorch-pretrained-bert
```
apex for mixed-precision training (optional)

# Download checkpoints

## X-LXMERT
```
wget -O x_lxmert/snap/pretrained/x_lxmert/Epoch20_LXRT.pth https://storage.googleapis.com/x_lxmert/Epoch20_LXRT.pth
```
## Image Generator
```
wget -O image_generator/pretrained/G_60.pth https://storage.googleapis.com/x_lxmert/G_60.pth
```

# Inference

Check out [inference/README.md](inference/README.md)

