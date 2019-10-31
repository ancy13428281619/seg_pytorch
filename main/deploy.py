from __future__ import print_function

import os
import torch
import cv2
import numpy as np
import torchvision
from model import getModels
from utils import utils
from torchvision import transforms
from torch.autograd import Variable
from toolbox import imgproctool as IPT


class Deployer:
    def __init__(self):
        # 加载配置文件
        self.config = utils.loadYaml('../config/config.yaml')
        # 训练结果保存路径
        self.output_model_path = os.path.join('../output/', self.config['Misc']['OutputFolderName'])
        # gpu使用
        self.device = utils.set_gpu(self.config)
        self.img_shape = (256, 256)
        self.overlap_piexl = 200
        self.imgs_index_dict, self.rois_start_xy_index_dict, self.rois_xyxy_index_dict = self.getRefInfo(
            ref_img_path='/home/pi/Desktop/df1b_dataset/20191024/ref_deploy')
        print(self.imgs_index_dict)
        print("Creating model")
        self.model = getModels(model_name=self.config['Model']['Name'],
                               num_classes=self.config['Model']['NumClass']).to(self.device)

        checkpoint = torch.load(os.path.join(self.output_model_path, self.config['Misc']['BestModelName']),
                                map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def overlap_crop(self, image, shape=(256, 256), overlap_row=0, overlap_col=0, divide_255=False):
        assert overlap_row < shape[0]
        assert overlap_col < shape[1]

        row, col = image.shape[:2]
        map = np.zeros(image.shape[:2], dtype=np.float32)

        if row <= shape[0] and col <= shape[1]:
            if divide_255:
                image = np.array(image, np.float32) / 255.0
            return [image], [[0, 0]], np.ones(image.shape[:2], dtype=np.float32)

        piece_list = []
        pts_list = []
        stride_row = shape[0] - overlap_row
        stride_col = shape[1] - overlap_col

        for row_n in range(0, row - shape[0] + stride_row, stride_row):
            for col_n in range(0, col - shape[1] + stride_col, stride_col):

                row_start = row_n
                row_end = row_n + shape[0]
                col_start = col_n
                col_end = col_n + shape[1]
                if row_n + shape[0] > row:
                    row_start = row - shape[0]
                    row_end = row
                if col_n + shape[1] > col:
                    col_start = col - shape[1]
                    col_end = col

                piece = image[row_start:row_end, col_start:col_end]
                map[row_start:row_end, col_start:col_end] += 1
                pts = [row_start, col_start]
                if divide_255:
                    piece = np.array(piece, np.float32) / 255.0
                piece_list.append(piece)
                pts_list.append(pts)

        return piece_list, pts_list, map

    def getRefInfo(self, ref_img_path):
        """
         根据ref获取相应视角下应保留的图片索引，即棱的地方
        :param ref_img_path:
        :return:
        """
        imgs_index_dict = {}
        rois_start_xy_list = []
        rois_xyxy_list = []

        imgs_name = os.listdir(ref_img_path)
        for img_name in imgs_name:
            if img_name.endswith('_mask.png'):

                box_type, _, view_point = img_name.split('_')[:3]
                mask = cv2.imread(os.path.join(ref_img_path, img_name))
                roi_imgs_list, rois_start_xy_list, rois_xyxy_list = self.overlap_crop(mask, shape=self.img_shape,
                                                                                      overlap_row=self.overlap_piexl,
                                                                                      overlap_col=self.overlap_piexl)
                rois_index_list = []

                for index, roi_img in enumerate(roi_imgs_list):
                    _, contours, _ = cv2.findContours(roi_img[..., 0].copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 1:
                        center_pt = np.mean(contours[0], axis=0, dtype=np.int32)[0]

                        if self.img_shape[0] // 4 < center_pt[0] < self.img_shape[0] // 4 * 3 and self.img_shape[
                            1] // 4 < center_pt[1] < self.img_shape[1] // 4 * 3:
                            # if cv2.countNonZero(roi_img[..., 0]) > 100:
                            rois_index_list.append(index)

                    # cv2.namedWindow('mask', 0)
                    # cv2.imshow('mask', roi_img)
                    # cv2.waitKey()
                imgs_index_dict[box_type + view_point] = rois_index_list

        return imgs_index_dict, rois_start_xy_list, rois_xyxy_list

    def preprocessImg(self, img):
        img_float = np.float32(img) / 255.
        img_tensor = torch.from_numpy(img_float)
        img_batch = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)

        img_batch = img_batch.to(self.device)
        return img_batch

    def run(self, root_path):
        # Data loading code
        print("Loading data")
        all_imgs_name = os.listdir(root_path)
        for img_name in all_imgs_name:
            if img_name.endswith('_1.png'):
                box_type, _, view_point = img_name.split('_')[:3]
                img_path = os.path.join(root_path, img_name)
                img = cv2.imread(img_path)
                roi_imgs_list, rois_start_xy_list, rois_xyxy_list = self.overlap_crop(img, shape=self.img_shape,
                                                                                      overlap_row=self.overlap_piexl,
                                                                                      overlap_col=self.overlap_piexl)
                for index, roi_img in enumerate(roi_imgs_list):
                    if index in self.imgs_index_dict[box_type + view_point]:
                        # cv2.imshow('img', roi_imgs_list[index])
                        # cv2.waitKey()

                        with torch.no_grad():
                            input_img = roi_imgs_list[index]
                            img_tensor = self.preprocessImg(input_img)
                            image = img_tensor.to(self.device, non_blocking=True)

                            output = self.model(image)
                            _, pred = output.topk(1, 1, largest=True, sorted=True)

                            if pred.cpu().numpy()[0][0] == 1:
                                xywh = (rois_start_xy_list[index][1], rois_start_xy_list[index][0], 256, 256)
                                IPT.drawRoi(img, xywh, IPT.ROI_TYPE_XYWH, (255, 255, 255), 2)
                                cv2.putText(input_img, 'ng', (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 0, 20), 1)
                            # cv2.imshow('roi', input_img)

                            # cv2.waitKey(33)
                cv2.namedWindow('img', 0)
                cv2.imshow('img', img)
                cv2.waitKey()


if __name__ == "__main__":
    deployer = Deployer()
    deployer.run(root_path='/home/pi/Desktop/df1b_dataset/20191024/src/ul/39')
