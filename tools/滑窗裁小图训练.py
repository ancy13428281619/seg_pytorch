#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np


def overlap_crop(image, shape=(256, 256), overlap_row=0, overlap_col=0, divide_255=False):
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


def getRefInfo(ref_img_path):
    """
     根据ref获取相应视角下应保留的图片索引，即棱的地方
    :param ref_img_path:
    :return:
    """
    imgs_index_dict = {}
    rois_start_xy_index_dict = {}
    rois_xyxy_index_dict = {}
    imgs_name = os.listdir(ref_img_path)
    for img_name in imgs_name:
        if img_name.endswith('_mask.png'):
            view_point = img_name.split('_')[2]
            mask = cv2.imread(os.path.join(ref_img_path, img_name))
            roi_imgs_list, rois_start_xy_list, rois_xyxy_list = overlap_crop(mask, shape=(256, 256), overlap_row=30,
                                                                             overlap_col=30)
            rois_index_list = []
            rois_start_xy_index_list = []
            rois_xyxy_index_list = []
            for index, roi_img in enumerate(roi_imgs_list):
                if cv2.countNonZero(roi_img[..., 0]) > 0:
                    rois_index_list.append(index)
                    rois_start_xy_index_list.append(index)
                    rois_xyxy_index_list.append(index)

                # cv2.namedWindow('mask', 0)
                # cv2.imshow('mask', roi_img)
                # cv2.waitKey()
            imgs_index_dict[view_point] = rois_index_list
            rois_start_xy_index_dict[view_point] = rois_start_xy_index_list
            rois_xyxy_index_dict[view_point] = rois_xyxy_index_list

    return imgs_index_dict, rois_start_xy_index_dict, rois_xyxy_index_dict


if __name__ == '__main__':
    # 所有图片的路径
    root_path = '/home/pi/Desktop/df1b_dataset/20191024/imgs'
    getRefInfo('/home/pi/Desktop/df1b_dataset/20191024/ref')
