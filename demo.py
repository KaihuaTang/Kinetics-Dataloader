#############################
#
#  Kaihua
#
#############################

import argparse
import torch
import numpy as np
import yaml
import utils_logger as logging
from loader import construct_loader
import utils_distributed as du


# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--anno_path', default=None, type=str, help='Indicate the path for annotation files.')

args = parser.parse_args()


# ============================================================================
# generate logger
logger = logging.custom_logger(output_path='./')


# ============================================================================
# load config
logger.info('=====> Load config from setting.yaml')
with open('setting.yaml') as f:
    if float(yaml.__version__) >= 5.4:
        config = yaml.full_load(f)
    else:
        config = yaml.load(f)


# ============================================================================
# upgrade config
config['data']['path_to_anno_dir'] = args.anno_path


# ============================================================================
# Set up environment.
du.init_distributed_training(config)
# Set random seed from configs.
np.random.seed(config['rng_seed'])
torch.manual_seed(config['rng_seed'])


# ============================================================================
# construct dataloader
train_loader = construct_loader(config, 'train', logger)
first_batch = train_loader[0]