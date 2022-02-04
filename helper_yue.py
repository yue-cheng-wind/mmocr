from mmcv import Config
# cfg = Config.fromfile('./configs/textrecog/sar/sar_r31_parallel_decoder_toy_dataset.py')
# cfg = Config.fromfile('./configs/textrecog/crnn/crnn_toy_dataset.py')
# cfg = Config.fromfile('./configs/textrecog/nrtr/nrtr_modality_transform_toy_dataset.py')
# cfg = Config.fromfile('./configs/textrecog/robust_scanner/robustscanner_r31_toy_dataset.py')
# cfg = Config.fromfile('./configs/textrecog/satrn/satrn_small_toy_dataset.py')
cfg = Config.fromfile('./configs/textrecog/abinet/abinet_academic_toy_dataset.py')



from mmdet.apis import set_random_seed
# Set up working dir to save files and logs.
cfg.work_dir = './demo/tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.001 / 8
cfg.lr_config.warmup = None
# Choose to log training results every 40 images to reduce the size of log file. 
cfg.log_config.interval = 40

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')


'''
Test build dataset
'''
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import os.path as osp

# Build dataset
datasets = [build_dataset(cfg.data.train)]



