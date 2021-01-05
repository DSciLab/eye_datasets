  
import numpy as np
from .base import BaseDataset
from .utils import transforms  as tf
from .utils.odir_parser import DataParser


class SubSet(object):
    def __init__(self):
        self.data = []
        self.len = None
        self.idx = None
    
    def init(self):
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def append(self, item):
        self.data.append(item)

    def __getitem__(self, index):
        if self.len == 0:
            return None
        index = index % self.len
        return self.data[index]


class Dataset(BaseDataset):
    def __init__(self, opt, data, validate=False):
        super(Dataset, self).__init__()
        self.validate = validate
        self.enabel_aug = opt.enabel_aug
        self.train_path = opt.train_path
        self.gen_path = opt.gen_path
        self.data = data
        self.num_classes = opt.num_classes
        self.sub_data = [SubSet() for _ in range(opt.num_classes)]
        self.dispatch()
        self.len = self.len * opt.num_classes

    def dispatch(self):
        for item in self.data:
            combo_label = item[4]
            for cls in combo_label:
                self.sub_data[cls].append(item)
        
        max_len = 0
        for i, subset in enumerate(self.sub_data):
            subset.init()
            print(f'classes {i}: data {len(subset)}')
            if len(subset) > max_len:
                max_len = len(subset)
        self.len = max_len
        print(f'max length {self.len}')

    def next_item(self, index):
        item = None
        while item is None:
            item = self.sub_data[index % self.num_classes][index // self.num_classes]
        return item

    def __getitem__(self, index):
        item = self.next_item(index)

        left_eye = item[0]
        right_eye = item[1]
        left_label = item[2]
        right_label = item[3]
        combo_label = item[4]

        left_image = self.read_npy(left_eye)
        right_image = self.read_npy(right_eye)

        left_image = self.random_transform(left_image)
        right_image = self.random_transform(right_image)

        left_image = tf.numpy_to_torch(left_image)
        right_image = tf.numpy_to_torch(right_image)

        left_label = self.one_hot(left_label)
        right_label = self.one_hot(right_label)
        combo_label = self.one_hot(combo_label)

        return left_image, left_label, right_image, right_label, combo_label


def get_data(opt):
    parser = DataParser(opt)
    data = []
    for item in parser.parse():
        data.append(item)
    return data
