import pickle
import random
import copy
import cv2
from torch.nn.modules.padding import ReflectionPad1d
from torchvision import transforms
from torch import nn


def image_to_binimage(image, threshold=10):
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    ret, binimage = cv2.threshold(gray_image, threshold, 255,
                                  cv2.THRESH_BINARY)
    return binimage


def get_roi(image, area_threshold=10000):
    bin_image = image_to_binimage(image)
    num_labels, labels, stats, centers = \
        cv2.connectedComponentsWithStats(bin_image, connectivity=8,
                                         ltype=cv2.CV_32S)

    roi_list = []
    num_labels_valid = 0
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        if area < area_threshold:
            continue
        num_labels_valid += 1
        roi_list.append((x, y, w, h))

    return num_labels_valid, roi_list


class Resize2D(object):
    def __init__(self, size):
        assert isinstance(size, int)
        self.x = size
        self.y = size

    def __call__(self, image):
        return cv2.resize(image, (self.x, self.y))


def read_meta(path):
    with open(path, 'rb') as f:
        meta = pickle.load(f)
    return meta


def dataset_split(data, train_ratio):
    data_len = len(data)
    train_len  = int(data_len * train_ratio)
    eval_len = data_len - train_len

    random.shuffle(data)
    train_data = data
    eval_data = copy.deepcopy(train_data[-eval_len:])
    del train_data[-eval_len:]

    return train_data, eval_data


class LinearNormalize(nn.Module):
    def forward(self, x):
        return (x - x.min()) / x.max()


normalize_fn = transforms.Normalize(mean=[0.473, 0.343, 0.263],
                                    std=[0.245, 0.183, 0.146])
linear_normalize_fn = LinearNormalize()
identity_normalize_fn = transforms.Lambda(lambda X: X)


def get_transform(opt):
    norm_opt = opt.get('normalize', 'linear')
    if norm_opt == 'linear':
        normalize = linear_normalize_fn
    elif norm_opt == 'identity':
        normalize = identity_normalize_fn
    else:
        normalize = normalize_fn

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(opt.image_size),
        transforms.RandomCrop(opt.image_size, 5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform