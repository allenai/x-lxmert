# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        if verbose:
            print("Optimizer: BERT")
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", action='store_true')

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size',
                        type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--tqdm", action='store_const',
                        default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss',
                        action='store_const', default=False, const=True)

    parser.add_argument('--backbone', type=str,
                        default='lxmert', help='lxmert|uniter')

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int,
                        help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int,
                        help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int,
                        help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa',
                        action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses',
                        default='obj,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument(
        "--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate',
                        default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const',
                        default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers',
                        default=0, type=int)

    # My configuration
    parser.add_argument("--grid_model", action='store_true')
    parser.add_argument('--n_boxes', type=int, default=36)
    parser.add_argument('--max_text_length', type=int, default=20)

    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    parser.add_argument("--grid_size", help='number of grids for image input crops',
                        default=8, type=int)

    parser.add_argument("--use_img_target", type=str2bool, default=False)

    parser.add_argument('--log_frequency', type=int, default=300,
                        help='log frequency')

    parser.add_argument("--COCOCaptionOnly",
                        dest="cococaptiononly", action='store_true')

    parser.add_argument("--dry", action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--mixed_precision", action='store_true')

    parser.add_argument("--visual_AR", action='store_true')

    parser.add_argument("--word_mask_predict", action='store_true')
    parser.add_argument("--vis_mask_predict", action='store_true')

    parser.add_argument("--cluster_src", type=str, default='mscoco_train')
    parser.add_argument("--clustering", action='store_true')
    parser.add_argument('--n_centroids', type=int, default=10000)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--n_class', type=int, default=1600)

    parser.add_argument("--feed_exact_feat", action='store_true')
    parser.add_argument("--target_exact_feat", action='store_true')
    parser.add_argument("--target_prob", action='store_true')

    parser.add_argument('--update_freq', type=int, default=1)

    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument("--v2", action='store_true')
    parser.add_argument("--v4", action='store_true')

    parser.add_argument('--start_epoch', type=int, default=None)

    parser.add_argument('--max_n_boxes', type=int, default=10)
    parser.add_argument('--n_max_single_class', type=int, default=7)

    parser.add_argument('--n_ratio', type=int, default=17)
    parser.add_argument('--n_scale', type=int, default=17)

    parser.add_argument('--n_classes', type=int, default=80)
    parser.add_argument('--n_max_distinct_classes', type=int, default=5)

    parser.add_argument('--n_iter_infer', type=int, default=10)

    parser.add_argument('--obj_p_threshold', type=float, default=0.5)

    parser.add_argument('--n_codebook', type=int, default=1000)
    parser.add_argument('--codebook_dim', type=int, default=64)
    parser.add_argument('--code_dot_product', action='store_true')

    parser.add_argument('--square_mask', action='store_true')
    parser.add_argument('--vis_all_mask', action='store_true')

    parser.add_argument('--VMP_smart', action='store_true')

    parser.add_argument('--code_regression', action='store_true')

    parser.add_argument('--encoder', type=str, default='resnet101')
    parser.add_argument('--vis_vocab_tune', action='store_true')
    parser.add_argument('--vis_vocab_from_scratch', action='store_true')
    parser.add_argument('--D_freeze_layers', type=int, default=-1)
    parser.add_argument('--D_tune', action='store_true')

    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--g_norm_type', type=str, default='spade_in')
    parser.add_argument('--gan', action='store_true')
    parser.add_argument('--hinge', action='store_true')
    parser.add_argument('--pixel_loss_lambda', type=float, default=0.0)
    parser.add_argument('--perceptual_loss_lambda', type=float, default=10.0)
    parser.add_argument('--gan_feat_match_lambda', type=float, default=10.0)
    parser.add_argument('--gan_feat_match_layers', type=int, default=-1)
    parser.add_argument('--gan_loss_lambda', type=float,
                        # default=0.01)
                        default=1.0)
    parser.add_argument('--resize_target_size', type=int, default=56)
    parser.add_argument('--resize_input_size', type=int, default=224)

    parser.add_argument('--sample_steps', type=int, default=10)
    parser.add_argument('--MP_out_intermediate', action='store_true')

    parser.add_argument('--sample_single_grid', action='store_true')
    parser.add_argument('--sample_AR', action='store_true')
    parser.add_argument('--sample_random', action='store_true')
    parser.add_argument('--sample_confidence', action='store_true')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--G_scheduled_sampling', action='store_true')

    parser.add_argument('--vis_sampling', action='store_true')
    parser.add_argument('--n_vis_sampling', type=int, default=100)

    parser.add_argument('--im_ratio', type=str, default='', choices=['', 'original'])

    parser.add_argument('--target_cluster', action='store_true')
    parser.add_argument('--target_obj_id', action='store_true')

    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--no_clip_grad', action='store_true')

    parser.add_argument('--caption_only', action='store_true')
    parser.add_argument('--vis_mask_COCO_only', action='store_true')
    parser.add_argument('--vis_mask_COCOVG_only', action='store_true')

    parser.add_argument('--comment', type=str, default='')

    # if parse and not is_interactive():
    if parse:
        # Parse the arguments.
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)

# if is_interactive():
#     args = None
# else:
#     args = parse_args()


if __name__ == '__main__':
    args = parse_args(True)
