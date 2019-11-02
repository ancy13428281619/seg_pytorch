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


class Tester(object):
    def __init__(self):
        # 加载配置文件
        self.config = utils.misc.loadYaml('../config/config.yaml')
        # gpu使用
        self.device = utils.misc.set_gpu(self.config)
        # 训练结果保存路径
        self.output_model_path = os.path.join('../output/', self.config['Misc']['OutputFolderName'])
        self.pred_color = (255, 0, 0)
        self.lable_color = (255, 0, 255)

    def test_one_epoch(self, dataset, model):
        model.eval()
        all_label_num = 0
        all_pred_num = 0
        all_fp = 0
        all_fn = 0
        for i, (img, mask) in enumerate(dataset):
            if i == len(dataset):
                break
            img_tensor = self.preprocess_image(img)
            pred = model.predict(img_tensor)
            pred = pred.squeeze().cpu().numpy().round()
            pred = pred.astype('uint8') * 255

            label_num, pred_num, fp, fn = self.get_test_result(img, mask, pred)
            all_label_num += label_num
            all_pred_num += label_num
            all_fp += fp
            all_fn += fn

        print("测试图片总数：{}".format(len(dataset)))
        print("标注总缺陷： {}".format(all_label_num))
        print("预测总缺陷： {}".format(all_pred_num))
        print("总fp： {}".format(all_fp))
        print("总fn： {}".format(all_fn))

    # 计算预测缺陷个数，误报，漏报
    def cal_fp_fn(self, img, mask, pred):
        # FP: 误报,2分类
        # FN: 漏报，2分类

        _, mask_contours, _, = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, pred_contours, _, = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img, mask_contours, -1, self.lable_color, 1)
        cv2.drawContours(img, pred_contours, -1, self.pred_color, 1)

        hit_n, label_num = self._cal_hit_num(mask, pred, mask_contours)
        hit_p, pred_num = self._cal_hit_num(mask, pred, mask_contours)

        fp = pred_num - hit_p
        fn = label_num - hit_n
        return label_num, pred_num, fp, fn

    def _cal_hit_num(self, mask, pred, contours):
        hit_num = 0
        for cm, _ in enumerate(contours):
            temp_m = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(temp_m, contours, cm, 255, thickness=-1)
            bl_p, bl_m = (pred == 255), (temp_m == 255)
            iou_and = np.sum(np.bitwise_and(bl_p, bl_m))
            if iou_and > 0:
                hit_num += 1
        return len(contours), hit_num

    def get_test_result(self, img, mask, pred):

        label_num, pred_num, fp, fn = self.cal_fp_fn(img, mask, pred)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "label: {}".format(label_num), (30, 40), font, 1.0, self.lable_color, 1)
        cv2.putText(img, "pred: {}".format(pred_num), (30, 80), font, 1.0, self.pred_color, 1)
        cv2.putText(img, "fn: {}".format(fn), (30, 120), font, 1.0, (0, 0, 255), 1)
        cv2.putText(img, "fp: {}".format(fp), (30, 160), font, 1.0, (0, 0, 255), 1)
        cv2.namedWindow('src', 0)
        cv2.imshow('src', img)
        # cv2.imshow('pred', pred)
        # cv2.imshow('label', mask)
        cv2.waitKey(33)
        return label_num, pred_num, fp, fn

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
    tester = Tester()
    tester.run()
