# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/3/26 19:52 
@Author : 弓长广文武
======================================
"""
import os
import random
import time
'''
======================================
@File    :   data.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2


# 重写数据集中的方法
class DatasetLoad(Dataset):
    def __init__(self, data_path, image_file, label_file, classes_num, image_mode, transforms=None):
        '''

        :param data_path: 数据文件目录
        :param image_file: 图像文在所在目录
        :param crop_file: 遮挡文在所在目录
        :param image_mode: 图像读取格式
        '''
        self.classes_num = classes_num
        # 获取数据路径
        self.data_path = data_path
        # 获取所有图像
        self.img_path = os.path.join(self.data_path, image_file)
        self.lab_path = os.path.join(self.data_path, label_file)
        # 获取该路径下的所有图像
        self.img_list = glob(os.path.join(self.img_path, '*.tif*'))
        self.lab_list = glob(os.path.join(self.lab_path, '*.tif*'))
        # 图像读取的模式 默认是1 RGB格式读取
        self.image_mode = image_mode
        # 图像增强变换
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [transforms]

    def __len__(self):
        # 返回训练集大小
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        lab_path = self.lab_list[index]
        # 读取影像
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), self.image_mode)
        lab = cv2.imdecode(np.fromfile(lab_path, dtype=np.uint8), 0)
        # 再次检查图像
        assert img.shape[:2] == lab.shape[:2], 'label与image图像大小不一致！'
        # 直接使用transforms进行图像变换
        # for transform in self.transforms:
        #     img = transform(img)
        #     lab = transform(lab)
        # 图像数值压缩
        img = img / 255.
        # 判断多分类还是二分类
        if self.classes_num < 3:
            lab = lab / 255.
            new_lab = np.zeros(lab.shape)
            # 标签数据转换为二值图像
            new_lab[lab > 0.5] = 1
            # new_lab[lab <= 0.5] = 0
            # new_lab = torch.from_numpy(new_lab).long()

        img = np.transpose(img, (2, 0, 1))
        return img, lab

class PredLoad(Dataset):
    def __init__(self, data_path, image_file, transforms):
        self.data_path = data_path
        # 获取所有图像
        self.img_path = os.path.join(self.data_path, image_file)
        # 获取该路径下的所有图像
        self.img_list = glob(os.path.join(self.img_path, '*.tif*'))
        # 图形变换增强
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

    def __len__(self):
        # 返回训练集大小
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        image_name = os.path.split(image_path)[1]
        # 读取影像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 处理图像
        # # 图像数值压缩
        # img = img / 255.
        # img = np.transpose(img, (2, 0, 1))
        # 直接使用transforms进行图像变换
        for transform in self.transforms:
            img = transform(img)
            # mask = mask / 255.
        return img, image_name

class PredOut(object):
    def __init__(self, pred_map, pred_save_path, pred_save_name, color_dict=None, classes_num=3, flag=''):
        self.pred_map = pred_map
        self.pred_save_path = pred_save_path
        self.pred_save_name = pred_save_name
        self.color_dict = color_dict
        self.classes_num = classes_num
        self.flag = flag

    def predprocess(self):
        for i, pred in enumerate(self.pred_map):
            pred_res = np.zeros(pred.shape, dtype=np.uint8)

            # 判断是多分类还是二分类
            if self.classes_num < 3:
                # 标签数据转换为二值图像
                pred_res[pred > 0.7] = 255
                # pred_res[pred <= 0.5] = 0

            else:
                for row in range(pred_res.shape[1]):
                    for col in range(pred_res.shape[2]):
                        index_of_class = np.argmax(pred[:, row, col])
                        pred_res[:, row, col] = self.color_dict[index_of_class]
                # pred_res = torch.argmax(pred, 1)

            name = self.pred_save_name[i]
            self.predsave(self.pred_save_path, name, pred_res)

    def predsave(self, path, name, result):
        now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
        name = os.path.splitext(name)[0] + self.flag + now_time + os.path.splitext(name)[1]
        if not self.classes_num < 3:
            pred_res = np.transpose(result, (1, 2, 0))
        else:
            pred_res = result[0, :, :]
        cv2.imencode('.tif', pred_res)[1].tofile(os.path.join(path, name))




if __name__ == "__main__":
    # isbi_dataset = DatasetLoad(r"E:\a学生文件\张广斌\data\my_data\CSD_S5\512\last_5000\no_en\train", 'image', 'label', 1, 0)
    # print("数据个数：", len(isbi_dataset))
    # train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
    #                                            batch_size=2,
    #                                            shuffle=True)
    # for image, label in train_loader:
    #     print(image.shape)
    image_path = r"E:\03work\老师分配\06、云雪检测\seq_snow_cloud\data\data_2021.01.29\1024\test\image\3.tif"
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    path = r"E:\03work\老师分配\06、云雪检测\seq_snow_cloud\data\data_2021.01.29\1024\test\image\10.tif"
    mask = gen_mask(img)
    c = crop(img, mask)
    cv2.imencode('.tif', c)[1].tofile(path)



# class BasicDataset(Dataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.scale = scale
#         self.mask_suffix = mask_suffix
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'
#
#         self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
#                     if not file.startswith('.')]
#         logging.info(f'Creating dataset with {len(self.ids)} examples')
#
#     def __len__(self):
#         return len(self.ids)
#
#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         w, h = pil_img.size
#         newW, newH = int(scale * w), int(scale * h)
#         assert newW > 0 and newH > 0, 'Scale is too small'
#         pil_img = pil_img.resize((newW, newH))
#
#         img_nd = np.array(pil_img)
#
#         if len(img_nd.shape) == 2:
#             img_nd = np.expand_dims(img_nd, axis=2)
#
#         # HWC to CHW
#         img_trans = img_nd.transpose((2, 0, 1))
#         if img_trans.max() > 1:
#             img_trans = img_trans / 255
#
#         return img_trans
#
#     def __getitem__(self, i):
#         idx = self.ids[i]
#         mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
#         img_file = glob(self.imgs_dir + idx + '.*')
#
#         assert len(mask_file) == 1, \
#             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
#         assert len(img_file) == 1, \
#             f'Either no image or multiple images found for the ID {idx}: {img_file}'
#         mask = Image.open(mask_file[0])
#         img = Image.open(img_file[0])
#
#         assert img.size == mask.size, \
#             f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
#
#         img = self.preprocess(img, self.scale)
#         mask = self.preprocess(mask, self.scale)
#
#         return {
#             'image': torch.from_numpy(img).type(torch.FloatTensor),
#             'mask': torch.from_numpy(mask).type(torch.FloatTensor)
#         }
#
#
# class CarvanaDataset(BasicDataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1):
#         super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')