

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
wget -O x-lxmert/snap/pretrained/x-lxmert/Epoch20_LXRT.pth https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/x-lxmert/Epoch20_LXRT.pth
```
## Image Generator
```
wget -O image_generator/pretrained/G_60.pth https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/image_generator/G_60.pth 
```

# Inference

Check out [./inference](inference/)
