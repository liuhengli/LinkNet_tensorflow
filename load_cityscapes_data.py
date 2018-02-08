# -*- coding: utf-8 -*-
import cv2
import pickle
import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
import random
"""
-------------------------------------------------
   File Name：     load_cityscapes_data
   version:        v1.0 
   Description :
   Author :       liuhengli
   date：          18-1-24
   license:        Apache Licence
-------------------------------------------------
   Change Activity:
                   18-1-24:
-------------------------------------------------
"""
__author__ = 'liuhengli'


def find_classes(root_dir):
    classes = ['Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
            'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain', 'Sky', 'Person',
            'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
    #classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(root_dir, mode):
    tensors = []
    data_dir = os.path.join(root_dir, 'leftImg8bit', mode)
    target_dir = os.path.join(root_dir, 'gtFine', mode)
    for folder in os.listdir(data_dir):
        d = os.path.join(data_dir, folder)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if filename.endswith('.png'):
                data_path = '{0}/{1}/{2}'.format(data_dir, folder, filename)
                target_file = filename.replace('leftImg8bit', 'gtFine_labelIds')
                target_path = '{0}/{1}/{2}'.format(target_dir, folder, target_file)
                item = (data_path, target_path)
                tensors.append(item)

    return tensors


def loader(input_path, target_path, new_img_width, new_img_height):
    input_image = cv2.imread(input_path, -1)
    input_image = cv2.resize(input_image, (new_img_width, new_img_height),
                           interpolation=cv2.INTER_NEAREST)
    target_image = cv2.imread(target_path, -1)
    target_image = cv2.resize(target_image, (new_img_width, new_img_height),
                           interpolation=cv2.INTER_NEAREST)
    # print(target_image.shape)
    # np.set_printoptions(threshold=np.NaN)
    # np.savetxt('imgae.txt', np.asarray(target_image, np.int32), fmt='%1.0e')
    # print(target_image)
    is_flip = random.choice([0, 1])
    if is_flip:
        input_image = cv2.flip(input_image, 1)
        target_image = cv2.flip(target_image, 1)

    return input_image, target_image


def remap_class():
    class_remap = {}
    class_remap[-1] = 0     #licence plate
    class_remap[0] = 0      #Unabeled
    class_remap[1] = 0      #Ego vehicle
    class_remap[2] = 0      #Rectification border
    class_remap[3] = 0      #Out of roi
    class_remap[4] = 0      #Static
    class_remap[5] = 0      #Dynamic
    class_remap[6] = 0      #Ground
    class_remap[7] = 1      #Road
    class_remap[8] = 2      #Sidewalk
    class_remap[9] = 0      #Parking
    class_remap[10] = 0     #Rail track
    class_remap[11] = 3     #Building
    class_remap[12] = 4     #Wall
    class_remap[13] = 5     #Fence
    class_remap[14] = 0     #Guard rail
    class_remap[15] = 0     #Bridge
    class_remap[16] = 0     #Tunnel
    class_remap[17] = 6     #Pole
    class_remap[18] = 0     #Polegroup
    class_remap[19] = 7     #Traffic light
    class_remap[20] = 8     #Traffic sign
    class_remap[21] = 9    #Vegetation
    class_remap[22] = 10    #Terrain
    class_remap[23] = 11    #Sky
    class_remap[24] = 12    #Person
    class_remap[25] = 13    #Rider
    class_remap[26] = 14    #Car
    class_remap[27] = 15    #Truck
    class_remap[28] = 16    #Bus
    class_remap[29] = 0     #Caravan
    class_remap[30] = 0     #Trailer
    class_remap[31] = 17    #Train
    class_remap[32] = 18    #Motorcycle
    class_remap[33] = 19    #Bicycle

    return class_remap


class SegmentedData:
    def __init__(self, root, mode, data_mode='small', start=0, end=1, new_img_width=1024, new_img_height=512, loader=loader):
        """
        Load data kept in folders ans their corresponding segmented data
        :param root: path to the root directory of data
        :type root: str
        :param mode: train/val mode
        :type mode: str
        :param transform: input transform
        :type transform: torch-vision transforms
        :param loader: type of data loader
        :type loader: function
        """
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, mode)

        self.data_mode = data_mode
        start = int(start * len(imgs))
        end = int(end * len(imgs))
        self.imgs = np.array(imgs[start:end])
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.new_img_width = new_img_width
        self.new_img_height = new_img_height
        self.loader = loader
        self.class_map = remap_class()
        self.id_to_trainId_map_func = np.vectorize(self.class_map.get)

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(imgs)
        print("The %s have total %d images" % (mode, self._num_examples))

    def reader(self, index):
        # Get path of input image and ground truth
        input_path, target_path = self.imgs[index]
        # Acquire input image and ground truth
        input_tensor, label_image = self.loader(input_path, target_path, self.new_img_width, self.new_img_height)
        # cv2.imshow("orign_label", label_image)
        if self.data_mode == 'small':
            # label_image.apply(lambda x: self.class_map[x])
            id_label = label_image
            output_label = self.id_to_trainId_map_func(id_label)
            # print(output_label)
            # np.savetxt('output_label.txt', np.asarray(output_label, np.int32), fmt='%1.1e')
            # print("=================")
            # idex = np.where(output_label>0)
            # print(idex)
            # print(np.max(output_label))

        return input_tensor, output_label

    def __len__(self):
        return len(self.imgs)

    def class_name(self):
        return (self.classes)

    def get_batch(self, batch_size):
        start = self._index_in_epoch
        images_batch = []
        labels_batch = []
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self.imgs = self.imgs[idx]  # get list of `num` random samples
            # self.anns = self.anns[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            for index in range(start, self._num_examples):
                image, label_image = self.reader(index)
                image = image.astype(np.float32) / 255.0
                images_batch.append(image)
                labels_batch.append(label_image)
            # img_rest_part = self.imgs[start:self._num_examples]
            # ann_rest_part = self.anns[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.imgs[idx0]  # get list of `num` random samples
            # self.anns = self.anns[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            for index in range(start, end):
                image, label_image = self.reader(index)
                image = image.astype(np.float32) / 255.0
                images_batch.append(image)
                labels_batch.append(label_image)
            # img_new_part = self.imgs[start:end]
            # ann_new_part = self.anns[start:end]
            labels_batch = np.asarray(labels_batch)
            labels_batch = np.reshape(labels_batch, (batch_size, self.new_img_height, self.new_img_width, 1))
            return images_batch, labels_batch
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            for index in range(start, end):
                image, label_image = self.reader(index)
                image = image.astype(np.float32) / 255.0
                images_batch.append(image)
                labels_batch.append(label_image)
            labels_batch = np.asarray(labels_batch)
            labels_batch = np.reshape(labels_batch, (batch_size, self.new_img_height, self.new_img_width, 1))
            return images_batch, labels_batch


if __name__ == '__main__':
    root_path = '/home/sdb/data/Cityscapes'
    data_obj_train = SegmentedData(root=root_path, mode='train', new_img_height=512, new_img_width=1024)
    for i in range(data_obj_train._num_examples):
        input_tensor, label_image = data_obj_train.reader(i)
        print(label_image.shape)
        cv2.imshow("orign", input_tensor)
        cv2.imshow("label", label_image.astype(np.uint8))
        cv2.waitKey(0)
