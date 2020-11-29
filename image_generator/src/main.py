from torchvision import transforms
import torch
import random
import numpy as np
import getpass
from pathlib import Path
from copy import deepcopy
from PIL import Image
import shutil
import logging
from datetime import datetime
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim

# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import apex
from apex import amp

import configs
from layers import Generator, Discriminator, ResNetEncoder]
from trainer import Trainer
from data_utils import get_loader


def main_worker(gpu, config):
    # GPU is assigned
    config.gpu = gpu
    config.rank = gpu
    print(f'Launching at GPU {gpu}')

    if config.distributed:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:9001',
                                # init_method="env://",
                                world_size=config.world_size,
                                rank=config.rank)

    if config.clustering:
        feat_dim = config.emb_dim # feat_dim
        imsize = config.resize_input_size # imsize
        n_centroids = config.n_centroids
        n_iter = config.n_iter
        encoder = config.encoder
        cluster_src = config.cluster_src

        centroid_dir = Path('../datasets/cluster_centroids/').resolve()
        if config.im_ratio == 'original':
            centroid_path = centroid_dir.joinpath(
                f'{encoder}_{cluster_src}_centroids{n_centroids}_iter{n_iter}_d{feat_dim}_grid{config.n_grid}.npy')
        else:
            centroid_path = centroid_dir.joinpath(
                f'{encoder}_{cluster_src}_centroids{n_centroids}_iter{n_iter}_d{feat_dim}_grid{config.n_grid}_imsize{imsize}.npy')
        centroids = np.load(centroid_path)

        Emb = nn.Embedding.from_pretrained(
            torch.from_numpy(centroids),
            freeze=True
            )
    else:
        Emb = None

    if config.classifier is None:
        E = None
    elif config.classifier == 'resnet101':
        E = ResNetEncoder('resnet101')
    elif config.classifier == 'resnet50':
        E = ResNetEncoder('resnet50')

    G = Generator(base_dim=config.g_base_dim,
                  emb_dim=config.emb_dim,
                  mod_dim=config.y_mod_dim,
                  n_channel=config.n_channel,
                  target_size=config.resize_target_size,
                  extra_layers=config.g_extra_layers,
                  init_H=config.n_grid,
                  init_W=config.n_grid,
                  norm_type=config.g_norm_type,
                  SN=config.SN,
                  codebook_dim=config.codebook_dim,
                  )

    if config.gan:
        D = Discriminator(base_dim=config.d_base_dim,
                          emb_dim=config.emb_dim,
                          n_channel=config.n_channel,
                          target_size=config.resize_target_size,
                          extra_layers=config.d_extra_layers,
                          init_H=config.n_grid,
                          init_W=config.n_grid,
                          SN=config.SN,
                          ACGAN=config.ACGAN,
                          n_classes=config.n_centroids
                          )
        if config.ACGAN:
            D.emb_classifier.weight = Emb.weight
    else:
        D = None

    # Logging
    if config.gpu == 0:
        logger = logging.getLogger('mylogger')
        file_handler = logging.FileHandler(config.log_dir.joinpath('log.txt'))
        stream_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)

        print('#===== (Trainable) Parameters =====#')

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        n_params = 0
        for model_name, model in [('E', E), ('G', G), ('D', D), ('Emb', Emb)]:
            if model is not None:
                # print(model)
                logger.info(model)
                # for name, p in model.named_parameters():
                #     print(name, '\t', list(p.size()))
                n_param = count_parameters(model)
                log_str = f'# {model_name} Parameters: {n_param}'
                print(log_str)
                logger.info(log_str)
                n_params += n_param
        log_str = f'# Total Parameters: {n_params}'
        logger.info(log_str)
        print(log_str)

        config.save(config.log_dir.joinpath('config.yaml'))

        # Save scripts for backup
        log_src_dir = config.log_dir.joinpath(f'src/')
        log_src_dir.mkdir(exist_ok=True)
        proj_dir = Path(__file__).resolve().parent
        for path in proj_dir.glob('*.py'):
            tgt_path = log_src_dir.joinpath(path.name)
            shutil.copy(path, tgt_path)
    else:
        logger = None

    if config.distributed:
        torch.cuda.set_device(config.gpu)

    if config.distributed:
        if 'bn' in config.g_norm_type:
            G = nn.SyncBatchNorm.convert_sync_batchnorm(G)
        G = G.cuda(config.gpu)

        params = G.parameters()

        g_optim = optim.Adam(params,
                             lr=config.g_lr,
                             betas=[config.g_adam_beta1,
                                    config.g_adam_beta2],
                             eps=config.adam_eps,
                             )

        if config.mixed_precision:
            G, g_optim = amp.initialize(G, g_optim, opt_level='O1')

        G = DDP(G, device_ids=[config.gpu],
                find_unused_parameters=True,
                broadcast_buffers=not config.SN)
    else:
        G = G.cuda()

        params = G.parameters()

        g_optim = optim.Adam(params,
                             lr=config.g_lr,
                             betas=[config.g_adam_beta1,
                                    config.g_adam_beta2],
                             eps=config.adam_eps,
                             )
        if config.multiGPU:
            G = nn.DataParallel(G)

    e_optim = None
    if config.classifier:
        if config.distributed:
            E = E.cuda(config.gpu)
        else:
            E = E.cuda()

        E = E.eval()
        if not config.distributed and config.multiGPU:
            E = nn.DataParallel(E)
    else:
        e_optim = None

    if config.gan:
        if config.distributed:
            D = D.cuda(config.gpu)

            d_optim = optim.Adam(D.parameters(),
                                 lr=config.d_lr,
                                 betas=[config.d_adam_beta1,
                                        config.d_adam_beta2],
                                 eps=config.adam_eps,
                                 )

            if config.mixed_precision:
                D, d_optim = amp.initialize(D, d_optim, opt_level='O1')

            D = DDP(D, device_ids=[config.gpu],
                    find_unused_parameters=True,
                    broadcast_buffers=not config.SN)
        else:
            D = D.cuda()

            d_optim = optim.Adam(D.parameters(),
                                 lr=config.d_lr,
                                 betas=[config.d_adam_beta1,
                                        config.d_adam_beta2],
                                 eps=config.adam_eps,
                                 )
            if config.multiGPU:
                D = nn.DataParallel(D)
    else:
        d_optim = None
    if config.clustering:
        if config.distributed:
            Emb = Emb.cuda(config.gpu)

        else:
            Emb = Emb.cuda()
            if config.multiGPU:
                Emb = nn.DataParallel(Emb)

    train_transform = transforms.Compose([
        transforms.Resize(
            (config.resize_input_size, config.resize_input_size),
            interpolation=Image.LANCZOS
        ),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(
            (config.resize_input_size, config.resize_input_size),
            interpolation=Image.LANCZOS
        ),
    ])

    data_out = ['img']
    if config.clustering:
        data_out.append('cluster_id')

    train_set = 'mscoco_train'
    if config.run_minival:
        train_set = 'mscoco_minival'

    train_loader = get_loader(
        config,
        train_set, mode='train', batch_size=config.batch_size,
        distributed=config.distributed, gpu=config.gpu,
        workers=config.workers, transform=train_transform,
        topk=config.train_topk, data_out=data_out)

    if config.distributed:
        valid_batch_size = config.batch_size
    else:
        valid_batch_size = config.batch_size // 4

    val_loader = get_loader(
        config,
        'mscoco_minival', mode='val', batch_size=valid_batch_size,
        distributed=config.distributed, gpu=config.gpu,
        workers=0, transform=valid_transform,
        topk=config.valid_topk, data_out=data_out)

    trainer = Trainer(config,
                      E, G, D, Emb,
                      g_optim, d_optim, e_optim,
                      train_loader, val_loader,
                      logger)
    trainer.train()


if __name__ == '__main__':

    config = configs.get_config()

    assert config.resize_target_size % config.n_grid == 0
    print(f'({config.n_grid} x {config.n_grid}) => ({config.resize_target_size} x {config.resize_target_size})')

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node

    current_time = datetime.now().strftime('%b%d_%H-%M')
    log_dir = Path(f'logs/{config.data}/{config.model}/').resolve()
    log_dir_name = current_time + f'_GPU{config.world_size}'
    log_dir_name += f'_{config.n_grid}_{config.resize_target_size}'

    if not config.ACGAN:
        log_dir_name += '_Proj'

    log_dir = log_dir.joinpath(log_dir_name)
    log_dir.mkdir(exist_ok=True, parents=True)
    log_str = str(config)
    log_str += '\nPyTorch Version: ' + torch.__version__
    log_str += '\nCurrent user: ' + getpass.getuser()
    log_str += f'\n# GPUs: {torch.cuda.device_count()}'
    log_str += '\nLogging at ' + str(log_dir)

    logger = logging.getLogger('mylogger')
    file_handler = logging.FileHandler(log_dir.joinpath('log.txt'))
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    # logger = None
    print(log_str)
    logger.info(log_str)

    config.log_dir = log_dir

    if config.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(config,))
    else:
        main_worker(0, config)
