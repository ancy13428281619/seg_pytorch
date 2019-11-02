import os
import cv2
import numpy as np
from .base_datalayer import BaseDataLayer


class Datalayer(BaseDataLayer):

    def __init__(self, config, augmentation=None, transform=None, target_transform=None):
        super(Datalayer, self).__init__()
        self.config = config
        train_dir = self.config['Dataset']['TrainPath']

        bg_imgs_dir = os.path.join(train_dir, 'bg')

        mask_suffix = '_mask.png'
        img_suffix = '.png'
        self.bg_imgs_path = [os.path.join(bg_imgs_dir, bg_img_name) for bg_img_name in os.listdir(bg_imgs_dir) if
                             bg_img_name.endswith(img_suffix)]

        ng_imgs_dir = os.path.join(train_dir, 'ng')
        self.ng_masks_path = [os.path.join(ng_imgs_dir, ng_img_name) for ng_img_name in os.listdir(ng_imgs_dir) if
                              ng_img_name.endswith(mask_suffix)]
        self.ng_imgs_path = [ng_mask_path.replace(mask_suffix, img_suffix) for ng_mask_path in self.ng_masks_path]

        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bg_imgs_path) + len(self.ng_masks_path)

    def __getitem__(self, item):
        # bg
        if np.random.random() > 0.5 and len(self.bg_imgs_path) > 0:
            # random_id_bg = np.random.randint(0, len(self.bg_imgs_path))
            random_id_bg = item % len(self.bg_imgs_path)
            img_path, mask_path = self.bg_imgs_path[random_id_bg], None
        # ng
        else:
            # random_id_ng = np.random.randint(0, len(self.ng_imgs_path))
            random_id_ng = item % len(self.ng_imgs_path)
            img_path, mask_path = self.ng_imgs_path[random_id_ng], self.ng_masks_path[random_id_ng]

        img = cv2.imread(img_path)

        mask = cv2.imread(mask_path, 0) if mask_path else np.zeros_like(img, dtype=np.uint8)[..., 0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask
