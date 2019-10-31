# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np


class Config:
    images_path = '/home/pi/Desktop/df1b_dataset/20191024/ref_deploy'
    roi_path = ''
    color = (255, 255, 255)
    mouse_pt = (0, 0)
    line_width = 20


def onMouse(event, x, y, flags, param):
    img = param[0]
    key_info = param[1]
    pts_list = param[2]

    Config.mouse_pt = (x, y)
    line_width = Config.line_width
    color = Config.color

    if event == cv2.EVENT_LBUTTONDOWN:
        if key_info == 'dot':
            cv2.circle(img, (x, y), line_width // 2, color, -1)
        elif key_info == 'line':
            pts_list.append((x, y))
        elif key_info == 'poly':
            pts_list.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        if key_info == 'brush':
            cv2.circle(img, (x, y), line_width // 2, color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if key_info == 'line':
            cv2.line(img, pts_list[0], (x, y), color, line_width)
            pts_list.clear()
        elif key_info == 'poly':
            if len(pts_list) > 1:
                cv2.line(img, pts_list[-2], pts_list[-1], color, 2)

    elif event == cv2.EVENT_MBUTTONDOWN:
        if len(pts_list) > 2:
            cv2.line(img, pts_list[0], pts_list[-1], color, 2)
            cv2.drawContours(img, np.array([pts_list]), -1, color, -1)
            pts_list.clear()


class LabelTool(object):
    def __init__(self):

        self.imgs_path_list, self.labels_path_list = self.getImagePaths()

        self.win_name = 'LabelTool'
        self.key_info = 'brush'
        self.pts_list = []
        self._createWins(self.win_name, False)
        self.img_index = 0
        self.img = None
        self.mask = None
        self.draw_img = None
        self.getImage()

    def getImagePaths(self):
        imgs_name = os.listdir(Config.images_path)
        labels_path = []
        imgs_path = []
        for img_name in imgs_name:
            if not img_name.endswith('_mask.png') and img_name.endswith('.png'):
                imgs_path.append(os.path.join(Config.images_path, img_name))
                labels_path.append(imgs_path[-1][:-4] + '_mask.png')
        return imgs_path, labels_path

    def _createWins(self, sWinName, bFullScreen):
        cv2.namedWindow(sWinName, 0)
        cv2.resizeWindow(sWinName, 1280, 960)
        cv2.moveWindow(sWinName, 0, 0)
        if bFullScreen:
            cv2.setWindowProperty(sWinName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def displayOperationInfo(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 0)
        cv2.putText(image, "name: {}".format(self.imgs_path_list[self.img_index].split('/')[-1]), (50, 50), font, 2.0,
                    color, 2)
        cv2.putText(image, "key 1: dot", (50, 100), font, 2.0, color, 2)
        cv2.putText(image, "key 2: line", (50, 150), font, 2.0, color, 2)
        cv2.putText(image, "key 3: brush", (50, 200), font, 2.0, color, 2)
        cv2.putText(image, "key 4: poly", (50, 250), font, 2.0, color, 2)
        cv2.putText(image, "key -: -", (50, 300), font, 2.0, color, 2)
        cv2.putText(image, "key +: +", (50, 350), font, 2.0, color, 2)
        self.draw_img = np.zeros_like(self.img)
        if self.key_info != 'poly':
            cv2.circle(self.draw_img, Config.mouse_pt, Config.line_width // 2, (255, 255, 0), 1)

    def responseKeyboard(self, key):

        if key == '1':
            self.key_info = 'dot'
        elif key == '2':
            self.key_info = 'line'
        elif key == '3':
            self.key_info = 'brush'
        elif key == '4':
            self.key_info = 'poly'
        elif key == '-':
            Config.line_width -= 5 if Config.line_width > 5 else 0
            # print('当前线宽： {}'.format(Config.line_width))
        elif key == '=':
            Config.line_width += 5 if Config.line_width < 500 else 0
            # print('当前线宽： {}'.format(Config.line_width))

    def getImage(self):
        self.img = cv2.imread(self.imgs_path_list[self.img_index])
        print('当前标注的图片的路径为： ', self.imgs_path_list[self.img_index], '序号：', self.img_index)

        self.draw_img = np.zeros_like(self.img)
        if not os.path.isfile(self.labels_path_list[self.img_index]):
            self.mask = np.zeros_like(self.img)
            cv2.imwrite(self.labels_path_list[self.img_index], self.mask)
        else:
            self.mask = cv2.imread(self.labels_path_list[self.img_index])

    def run(self):
        while True:
            key = chr(cv2.waitKey(33) & 255).lower()
            self.responseKeyboard(key)
            if key == 'q':
                break
            elif key == 'c':
                self.mask = np.zeros_like(self.img)
            elif key == 'n':
                if self.img_index < len(self.imgs_path_list):
                    cv2.imwrite(self.labels_path_list[self.img_index], self.mask)
                    self.img_index += 1
                    self.getImage()
                else:
                    break
            elif key == 'f':
                self._createWins(self.win_name, True)
            elif key == 'p':
                self._createWins(self.win_name, False)
            self.displayOperationInfo(self.img)
            cv2.setMouseCallback(self.win_name, onMouse, [self.mask, self.key_info, self.pts_list])
            display_img = cv2.bitwise_or(self.img, self.mask)
            display_img = cv2.bitwise_or(display_img, self.draw_img)
            cv2.imshow(self.win_name, display_img)


if __name__ == '__main__':
    label_tool = LabelTool()
    label_tool.run()
