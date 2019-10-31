from __future__ import print_function

import os
import torch
import cv2
import numpy as np
import torchvision
from model import getModels
from utils import utils
from torchvision import transforms
from datalayer.datalayer import Datalayer
import shutil

class Tester:
    def __init__(self):
        # 加载配置文件
        self.config = utils.loadYaml('../config/config.yaml')
        # 训练结果保存路径
        self.output_model_path = os.path.join('../output/', self.config['Misc']['OutputFolderName'])
        # gpu使用
        self.device = utils.set_gpu(self.config)

    def run(self):
        # Data loading code
        print("Loading data")
        test_dir = self.config['Dataset']['TestPath']
        pred_result_dir = os.path.join(test_dir, 'pred_result_info')
        utils.mkdir(pred_result_dir)

        dataset_test = torchvision.datasets.ImageFolder(test_dir,
                                                        transforms.Compose([
                                                            transforms.ToTensor()]))
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=1, pin_memory=True)

        print("Creating model")
        model = getModels(model_name=self.config['Model']['Name'], num_classes=self.config['Model']['NumClass'])
        model.to(self.device)

        checkpoint = torch.load(os.path.join(self.output_model_path, self.config['Misc']['BestModelName']),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        correct_1 = 0.0
        model.eval()
        print(dataset_test.class_to_idx)
        with torch.no_grad():
            for i, (image, label) in enumerate(data_loader_test):
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                output = model(image)

                _, pred = output.topk(1, 1, largest=True, sorted=True)
                label = label.view(label.size(0), -1).expand_as(pred)

                correct = pred.eq(label)
                correct_1 += correct[:, :1].sum()

                for key in dataset_test.class_to_idx:
                    if dataset_test.class_to_idx[key] == label:
                        label_info = key
                    if dataset_test.class_to_idx[key] == pred:
                        pred_info = key

                src_img = cv2.imread(dataset_test.imgs[i][0], 1)
                cv2.putText(src_img, 'label: {}'.format(label_info), (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 1)
                cv2.putText(src_img, 'pred: {}'.format(pred_info), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 20), 1)
                if label_info != pred_info:
                    shutil.move(dataset_test.imgs[i][0], pred_result_dir)
                    # cv2.imwrite(os.path.join(pred_result_dir, '{}.png'.format(i)), src_img)
                cv2.imshow('img', src_img)
                cv2.waitKey(33)

        print("acc: ", correct_1 / len(data_loader_test.dataset))


if __name__ == "__main__":
    tester = Tester()
    tester.run()
