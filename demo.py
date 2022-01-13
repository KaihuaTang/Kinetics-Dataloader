#############################
#
#  Kaihua
#
#############################

import argparse
import yaml
import utils_logger as logging


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
    config = yaml.load(f)


# ============================================================================
# upgrade config
config['data']['path_to_anno_dir'] = args.anno_path



