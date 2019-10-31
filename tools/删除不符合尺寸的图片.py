import os
import cv2
import shutil

root_path = '/media/ancy/文档/AI/datasets/df1b_data/small_img/'
ng_path = os.path.join(root_path, 'ng')
bg_path = os.path.join(root_path, 'bg')

input_shape = [256, 256]

all_files = os.listdir(ng_path)
for img_name in all_files:

    if cv2.imread(os.path.join(ng_path, img_name), 0).shape != (256, 256):

        print(os.path.join(ng_path, img_name))
        os.remove(os.path.join(ng_path, img_name))
