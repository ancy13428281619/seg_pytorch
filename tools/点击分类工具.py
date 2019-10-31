# coding=utf-8
import random

import cv2
import numpy as np
import os, shutil


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        try:
            shutil.move(srcfile, dstfile)  # 移动文件
            print("move %s -> %s" % (srcfile, dstfile))
        except:
            print('warning! no this file!')


def merge_img(images):
    l = len(images) // 3
    img_up = np.hstack(images[:l])
    img_middle = np.hstack(images[l:l * 2])
    img_down = np.hstack(images[l * 2:])
    img = np.vstack([img_up, img_middle, img_down])
    return img


def on_mouse(event, x, y, flags, param):
    global name_list, move_dir, merge, img_shape

    if event == cv2.EVENT_LBUTTONDOWN:
        l = len(name_list) // 3
        id = x // img_shape[1] + (y // img_shape[0]) * l
        cv2.putText(merge, str(id) + ':', (x - 100, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 2)
        cv2.putText(merge, os.path.split(name_list[id])[-1], (x, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 2)
        print(id)
        cv2.imshow('image', merge)
        mymovefile(name_list[id], move_dir)
        mymovefile(name_list[id][:-4] + '_mask.png', move_dir)


def batch_cls_imgs(img_path, all_l=9):
    global name_list, move_dir, merge, img_shape
    move_dir = img_path + '/点击挑选出来的/'

    list = os.listdir(img_path)
    img_list = []
    for i in range(0, len(list)):
        path = os.path.join(img_path, list[i])
        if os.path.isfile(path) and \
                path.endswith('.png') is True and \
                path.endswith('mask.png') is False and \
                path.endswith('pred.png') is False and \
                path.endswith('display.png') is False:
            img_list.append(i)

    img_list = sorted(img_list)
    print(img_list)
    i = 0
    while True:
        temp_list = []
        name_list = []
        for j in range(all_l):
            image_path = os.path.join(img_path, list[img_list[i]])
            image = cv2.imread(image_path)
            if image is None:
                image = np.zeros(img_shape, dtype=np.uint8)
            img_shape = image.shape
            temp_list.append(image)
            name_list.append(image_path)
            i += 1
        print(i)

        merge = merge_img(temp_list)

        cv2.namedWindow('image', 0)
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', merge)

        key = chr(cv2.waitKeyEx(0) & 255)

        if key in ['z', 'Z']:
            i = i - all_l * 2


if __name__ == '__main__':
    # 将文件分为ok和ng
    # cls_good_ng()
    # 批量分类
    batch_cls_imgs(img_path='/home/pi/Desktop/df1b_dataset/20191024/small_img_train/bg', all_l=21)
