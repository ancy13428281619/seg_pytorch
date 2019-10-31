# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np


class CropBGNGByMaskRef(object):
    def __init__(self):
        root_path = '/media/ancy/文档/AI/datasets/df1b_data/imgs/train/ch_1'
        ref_path = '/media/ancy/文档/AI/datasets/df1b_data/ref'
        self.bg_output_path = '/media/ancy/文档/AI/datasets/df1b_data/small_img/bg'
        self.ng_output_path = '/media/ancy/文档/AI/datasets/df1b_data/small_img/ng'

        self.img_shape = [256, 256]
        self.imgs_path_list, self.masks_path_list = self.getImgMaskPath(root_path)
        self.ref_imgs_path_dict = self.getRefImgsPath_dict(ref_path)
        self.img_counter = 0

    def getImgMaskPath(self, imgs_path):
        # 图片路径提取
        imgs_path_list = []
        masks_path_list = []
        for img_name in os.listdir(imgs_path):
            if img_name.endswith('_1_mask.png'):
                box_type, _, view_point = img_name.split('_')[0:3]
                masks_path_list.append(os.path.join(imgs_path, img_name))
                imgs_path_list.append(os.path.join(imgs_path, img_name[:-9] + '.png'))
        return imgs_path_list, masks_path_list

    def getRefImgsPath_dict(self, ref_imgs_path):
        ref_imgs_path_dict = {}
        for ref_img_name in os.listdir(ref_imgs_path):
            if ref_img_name.endswith('_mask.png'):
                box_type, _, view_point = ref_img_name.split('_')[0:3]
                ref_imgs_path_dict[box_type + view_point] = os.path.join(ref_imgs_path, ref_img_name)
        return ref_imgs_path_dict

    def getRoiImg(self, img, center_pt, roi_size):
        # print(center_pt, roi_size)
        if center_pt[0] - roi_size[0] // 2 < 0:
            center_pt[0] = roi_size[0] // 2 + 1
        if center_pt[1] - roi_size[1] // 2 < 0:
            center_pt[1] = roi_size[1] // 2 + 1
        if center_pt[0] + roi_size[0] // 2 > img.shape[0]:
            center_pt[0] = img.shape[0] - roi_size[0] // 2 + 1
        if center_pt[1] + roi_size[1] // 2 > img.shape[1]:
            center_pt[1] = img.shape[1] - roi_size[1] // 2 + 1

        return img[center_pt[1] - roi_size[1] // 2:center_pt[1] + roi_size[1] // 2,
               center_pt[0] - roi_size[0] // 2:center_pt[0] + roi_size[0] // 2]

    def cropNgImg(self, crop_img_num=10):
        for img_index in range(len(self.imgs_path_list)):
            img = cv2.imread(self.imgs_path_list[img_index])
            mask = cv2.imread(self.masks_path_list[img_index], 0)

            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(crop_img_num):
                for contour in contours:
                    offset_pixel = np.random.randint(0, self.img_shape[0] // 4)
                    center_pt = np.mean(contour, axis=0, dtype=np.int32)[0]
                    # print(cv2.contourArea(contour))

                    center_pt += offset_pixel
                    mask_roi = self.getRoiImg(mask, center_pt, self.img_shape)
                    # 靠近图像边缘20像素不能有缺陷
                    mask_roi_small = self.getRoiImg(mask, center_pt, np.array(self.img_shape) - 20)
                    if cv2.countNonZero(mask_roi_small) > 0 and cv2.countNonZero(mask_roi) == cv2.countNonZero(
                            mask_roi_small):
                        cv2.imwrite(os.path.join(self.ng_output_path, '{}.png'.format(self.img_counter)),
                                    self.getRoiImg(img, center_pt, self.img_shape))
                        self.img_counter += 1
                    # cv2.circle(img, tuple(center_pt), 3, (255, 255, 0), -1)
                    #
                    # cv2.imshow('mask', mask_roi)
                    # cv2.imshow('mask_0', mask_roi_0)

            # cv2.namedWindow('img', 0)
            #
            # cv2.imshow('img', img)
            # cv2.waitKey()

    def cropBgImg(self, random_num=20):
        for i, img_path in enumerate(self.imgs_path_list):
            img_name = img_path.split('/')[-1]
            box_type, _, view_point = img_name.split('_')[0:3]
            ref_img = cv2.imread(self.ref_imgs_path_dict[box_type + view_point])[..., 2]
            img = cv2.imread(img_path)
            mask = cv2.imread(self.masks_path_list[i], 0)
            for i in range(random_num):
                y_list, x_list = np.where(ref_img > 100)
                random_pt_index = np.random.randint(0, len(y_list), dtype=np.int64)

                roi_img = self.getRoiImg(img, [x_list[random_pt_index], y_list[random_pt_index]], self.img_shape)
                roi_mask = self.getRoiImg(mask, [x_list[random_pt_index], y_list[random_pt_index]], self.img_shape)
                if cv2.countNonZero(roi_mask) == 0:
                    cv2.imwrite(os.path.join(self.bg_output_path, '{}.png'.format(self.img_counter)), roi_img)
                    self.img_counter += 1

    def run(self):
        self.cropNgImg(5)
        self.cropBgImg(5)


if __name__ == '__main__':
    cb = CropBGNGByMaskRef()
    cb.run()
