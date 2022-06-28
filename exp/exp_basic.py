import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.chkpath = None

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            # self.logger.info('| Use GPU: cuda:{}'.format(self.args.gpu))
            self.logger.info('| Use GPU')
        else:
            device = torch.device('cpu')
            self.logger.info('| Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self, settings):
        pass

    def test(self):
        pass
