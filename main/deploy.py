from __future__ import print_function
import os
import sys

_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_FILE_PATH, "../"))
import torch
import utils.misc
import utils.metrics
import numpy as np
from datalayer.datalayer import Datalayer
from model import getModels
import cv2


class Deployer(object):
    def __init__(self):
        # 加载配置文件
        self.config = utils.misc.loadYaml('../config/config.yaml')
        # gpu使用
        self.device = utils.misc.set_gpu(self.config)
        # 训练结果保存路径
        self.output_model_path = os.path.join('../output/', self.config['Misc']['OutputFolderName'])

    def test_one_epoch(self, dataset, model):
        model.eval()
        for i, (img, mask) in enumerate(dataset):
            if i == len(dataset):
                break
            img_tensor = self.preprocess_image(img)
            pred = model.predict(img_tensor)
            pred = pred.squeeze().cpu().numpy().round()
            cv2.imshow('src', img)
            cv2.imshow('pred', pred)
            cv2.imshow('label', mask)
            cv2.waitKey()

    def preprocess_image(self, img):
        img_float = np.float32(img) / 255.
        img_tensor = torch.from_numpy(img_float)
        img_batch = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        img_batch = img_batch.cuda() if torch.cuda.is_available() else img_batch
        return img_batch

    def run(self):
        # 创建模型
        model, preprocessing_fn = getModels(self.config)
        model = model.to(self.device)
        # 加载数据
        dataset = Datalayer(self.config)
        # 加载权重
        checkpoint = torch.load(os.path.join(self.output_model_path, self.config['Misc']['BestModelName']),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # 开始测试
        self.test_one_epoch(dataset, model=model)


if __name__ == "__main__":
    deployer = Deployer()
    deployer.run()
