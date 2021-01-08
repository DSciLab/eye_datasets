import pickle
from .base import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, data, train=True, transform=None):
        super(Dataset, self).__init__()
        self.train = train
        self.trainsform =transform
        self.data = data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_pil(path):
        with open(path, 'rb') as f:
            pil_image = pickle.load(f)
        return pil_image

    def __getitem__(self, index):
        item = self.data[index]

        image = item[0]
        label = item[1]

        image = self.load_pil(image)

        if self.trainsform is not None:
            image = self.trainsform(image)

        return image, label
