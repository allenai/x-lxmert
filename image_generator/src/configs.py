import argparse
import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def read_config(path):
    return Config.load(path)


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--g_lr', type=float, default=0.0004,
                        help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for D')
    parser.add_argument('--g_adam_beta1', type=float, default=0.0,
                        help='Adam beta 1 for G')
    parser.add_argument('--g_adam_beta2', type=float, default=0.999,
                        help='Adam beta 2 for G')
    parser.add_argument('--d_adam_beta1', type=float, default=0.0,
                        help='Adam beta 1 for D')
    parser.add_argument('--d_adam_beta2', type=float, default=0.999,
                        help='Adam beta 2 for D')
    parser.add_argument('--adam_eps', type=float, default=1e-7,
                        help='Adam eps')

    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='gradient clipping')

    parser.add_argument('--multiGPU', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')

    parser.add_argument('--dry', action='store_true',
                        help='Run training script without actually running training steps. Debugging only')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--deterministic', type=str2bool, default=True)

    parser.add_argument('--tqdm_len', type=int, default=280,
                        help='tqdm log print length')
    parser.add_argument('--workers', type=int, default=4,
                        help='training loader workers')

    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--log_img_step', type=int, default=100)


    # Model
    parser.add_argument('--model', type=str, default='ResnetGridSpade')

    parser.add_argument('--g_base_dim', type=int, default=64)
    parser.add_argument('--d_base_dim', type=int, default=64)

    parser.add_argument('--g_extra_layers', type=int, default=0)
    parser.add_argument('--d_extra_layers', type=int, default=0)

    parser.add_argument('--g_norm_type', type=str, default='spade_in', help='default: SPADE version of Instance Norm')
    parser.add_argument('--emb_dim', type=int, default=2048)
    parser.add_argument('--y_mod_dim', type=int, default=128)
    parser.add_argument('--codebook_dim', type=int, default=256)

    parser.add_argument('--SN', action='store_true', help='whether to use spectral norm')

    parser.add_argument('--clustering', action='store_true')

    parser.add_argument('--classifier', type=str, default='resnet50', help='Pretrained classifier (Perceptual loss)')


    # Loss
    parser.add_argument('--gan', action='store_true', help='use gan loss')
    parser.add_argument('--hinge', action='store_true', help='use hinge version gan loss')
    parser.add_argument('--ACGAN', action='store_true', help='ACGAN style CGAN loss. If False, then use Projection Discriminator')

    parser.add_argument('--recon_loss_lambda',  type=float, default=0.0)
    parser.add_argument('--obj_loss_lambda',  type=float, default=0.0)
    parser.add_argument('--feat_loss_lambda',  type=float, default=10.0)

    parser.add_argument('--all_layers', action='store_true')
    parser.add_argument('--feat_pool', type=str2bool, default=False)

    parser.add_argument('--gan_feat_match_lambda', type=float, default=10.0)
    parser.add_argument('--gan_feat_match_layers', type=int, default=-1)
    parser.add_argument('--gan_loss_lambda', type=float, default=1.0)
    parser.add_argument('--gan_loss_cluster_lambda', type=float, default=1.0)

    # Data
    parser.add_argument('--data', type=str, default='mscoco')
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=1000)

    parser.add_argument('--run_minival', action='store_true')
    parser.add_argument('--n_channel', type=int, default=3)

    parser.add_argument('--resize_target_size', type=int, default=256)
    parser.add_argument('--resize_input_size', type=int, default=512)
    parser.add_argument('--n_grid', type=int, default=8)

    # Clustering
    parser.add_argument('--encoder', type=str, default='maskrcnn')
    parser.add_argument('--n_centroids', type=int, default=10000)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--cluster_src', type=str, default='mscoco_train')
    parser.add_argument('--im_ratio', type=str, default='original')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
