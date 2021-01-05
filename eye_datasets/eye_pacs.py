import numpy as np
import torch
import pandas as pd
import os

from .base import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, opt, data, validate=False, test=False):
        super(Dataset, self).__init__()
        title = 'Train'
        if test:
            title = 'Test'
        if validate:
            title = 'Validate'
        print(f'================ Total {title} Data {len(data)} ====================')

        self.validate = validate
        self.test = test
        self.enable_exclusive = opt.enable_exclusive
        self.enable_aug = opt.enable_aug
        if not test:
            self.image_root_path = opt.train_root
        else:
            self.image_root_path = opt.test_root
        self.len = len(data)
        self.data = data
        self.num_classes = opt.num_classes
        self.eye = torch.eye(self.num_classes)

    def one_hot(self, label):
        return self.eye[label]

    def read_npy(self, image_name):
        npy_name = f'{image_name}.npy'
        image_npy_path = os.path.join(self.image_root_path, npy_name)
        image = np.load(image_npy_path)
        return image

    def __getitem__(self, index):
        if self.test or self.validate:
            #########################################################
            #                 Test or Validate                      #
            #########################################################
            item = self.data[index]
            if not self.test:
                # For validate
                image_name = item[0]
                label = item[1]

                image = self.read_npy(image_name)
                image = self.random_transform(image)
                image = tf.numpy_to_torch(image)
                label = self.one_hot(label)
                return image, label
            else:
                # For test
                image_name = item
                image = self.read_npy(image_name)
                image = self.random_transform(image)
                image = tf.numpy_to_torch(image)
                return image_name, image

        #########################################################
        #                     Training                          #
        #########################################################
        item = self.data[index]
        image = item[0]
        label = item[1]

        image = self.read_npy(image)
        image = self.random_transform(image)
        image = tf.numpy_to_torch(image)
        label = self.one_hot(label)

        return index, image, label


def read_csv(path):
    data = pd.read_csv(path, sep=',')
    return data


def get_data(opt):
    print('Get drd data')
    df = read_csv(opt.annotation)
    data = []
    for i in range(len(df)):
        image_name = df.iloc[i]['image']
        label = df.iloc[i]['level']
        data.append([image_name, label])
    return data


def get_test_data(opt):
    tabel = read_csv(opt.test_tabel)
    tabel_len = len(tabel)
    data = []

    for i in range(tabel_len):
        image_name = tabel.iloc[i]['image']
        data.append(image_name)
    return data
