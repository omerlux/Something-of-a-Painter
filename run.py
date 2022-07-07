import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
from exp.exp_main import Exp_Main
from utils.tools import create_exp_dir
import wandb

parser = argparse.ArgumentParser(description='GANs for Monet photos generator')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--wandb', action='store_true', help='activating wandb updates')
parser.add_argument('--save', type=str, default='TMP', help='name of main directory')
parser.add_argument('--model_id', type=str, default='EXP', help='model id')
parser.add_argument('--model', type=str, default='CycleGan',
                    help='model name, options: [CycleGan]')
parser.add_argument('--seed', type=int, default=2021, help='seed number')

# data loader
parser.add_argument('--data', type=str, default='IGDK', help='dataset type')
parser.add_argument('--root_path', type=str, default='../../data/Image_Generation_Data_Kaggle/',
                    help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='location of model checkpoints')
parser.add_argument('--ds_augment', action='store_true',
                    help='pre-augmentation to the datasets')
parser.add_argument('--diffaugment', nargs="+", type=str, default=[],
                    help='kind of DiffAugmentation to apply ["color", "translation", "cutout"]')

# model and task parameters task
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--transformer_blocks', type=int, default=6)
parser.add_argument('--cycle_noise', type=float, default=0, help='cycle std noise added to generated image. 0 is none')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--log_interval', type=int, default=20, help='training log print interval')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--steps_per_epoch', type=int, default=-1, help="steps per epoch, -1 is max(monet_ds, photo_ds)/4")
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# TODO: not working right now
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer name, options: [adam, rmsprop]')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu - if mentioned in args, no gpu')
parser.add_argument('--devices', type=str, default="0", help='device ids of multile gpus')

args = parser.parse_args()
args.auto = tf.data.experimental.AUTOTUNE
gettrace = getattr(sys, 'gettrace', lambda: None)
args.debug = gettrace() is not None

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# accessibility to gpus
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # tf.config.run_functions_eagerly(False)  # or True
    try:
        for gpu in gpus:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = False
            config.gpu_options.per_process_gpu_memory_fraction = 0.95
            session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(session)
            # tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if args.is_training:
    if args.wandb:
        wandb.init(project="painter", entity="omerlux", config=args,
                   tags=["Debug" if args.debug else "Exp", args.data, args.model])
        config = wandb.config

    # setting a folder to current run
    if args.save == 'TMP':
        args.save = args.data + '-' + args.model
        if args.debug:
            args.save += '-DEBUG'
        else:
            args.save += '-EXP'

        if args.model_id != 'EXP':
            args.save += '-' + args.model_id

        args.save = 'saves/{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

    if 'ResCycleGan' in args.model:
        model_script = 'models/CycleGan.py'
    else:
        model_script = 'models/CycleGan.py'

    main_script = 'exp/exp_main.py'
    if os.path.isdir(os.path.join('saves', args.save)):
        args.save = os.path.join('saves', args.save)
    else:
        create_exp_dir(args.save, scripts_to_save=[model_script, main_script])

else:
    args.save = os.path.join('saves', args.save)

# setting logger
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# if args.data_path == '': args.data_path = args.data + '.csv'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    logging.info('| GPU devices: cuda:{}'.format(device_ids))
    args.device_ids = list(range(len(device_ids)))  # [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

Exp = Exp_Main

if args.is_training:
    logging.info('| >>> Args in experiment:')
    logging.info('| {}'.format(args))
    for ii in range(args.itr):
        logging.info('=' * 80)
        # setting record of experiments
        setting = '{}_{}_{}_sd{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seed,
            ii
        )

        exp = Exp(args, logging)  # set experiments
        logging.info('=========== Start training : {} ==========='.format(setting))
        exp.train(setting)

        if args.do_predict:
            logging.info('=========== Predicting : {} ==========='.format(setting))
            exp.predict(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_sd{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.seed,
        ii
    )

    exp = Exp(args, logging)  # set experiments
    logging.info('=========== Predicting : {} ==========='.format(setting))
    exp.predict(setting)
    torch.cuda.empty_cache()
