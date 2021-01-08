import torch
import pickle
from .base import BaseDataset


class Dataset(BaseDataset):
    num_classes = 8

    def __init__(self, data, train=True,
                 transform=None):
        super(Dataset, self).__init__()
        self.train = train
        self.transform = transform
        self.data = data
        self.eye = torch.eye(self.num_classes)
    
    def one_hot(self, label):
        result = torch.zeros(8)
        for i in label:
            result += self.eye[i]
        return result

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_pil(path):
        with open(path, 'rb') as f:
            pil_image = pickle.load(f)
        return pil_image

    def __getitem__(self, index):
        item = self.data[index]

        age = item['age']
        sex = item['sex']
        left_eye = item['left_eye']
        right_eye = item['right_eye']
        left_label = item['left_label']
        right_label = item['right_label']
        combo_label = item['combo_label']

        left_image = self.load_pil(left_eye)
        right_image = self.load_pil(right_eye)

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        left_label = self.one_hot(left_label)
        right_label = self.one_hot(right_label)
        combo_label = self.one_hot(combo_label)

        return age, sex, left_image, left_label, right_image, right_label, combo_label
