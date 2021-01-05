  
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import transforms  as tf


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.eye = torch.eye(8)
        self.color_jitter = tf.ColorJitter(brightness=1.0, contrast=1.0,
                                           saturation=0.3, hue=0.1)

    def one_hot(self, label):
        result = torch.zeros(8)
        for i in label:
            result += self.eye[i]
        return result

    def __len__(self):
        return self.len

    def random_transform(self, image):
        if not self.validate and not self.test:
            if np.random.rand() >= 0.5:
                image = tf.Affine.reflect(image)

            if self.enable_aug:
                angle = np.random.normal(0, 2) * 360
                image = tf.Affine.rotate(image, angle, reshape=False)
                image = tf.Affine.scale(image, std=0.03, keep_size=True)
                # image = tf.Affine.translation(image, std=0.03)
                # image = tf.Affine.randColor(image, std=0.3)
                image = self.color_jitter(image)
        return image
