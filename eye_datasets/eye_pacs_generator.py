import glob
import os
from PIL import Image
import pandas as pd
import shutil
import pickle
import numpy as np
from pandas.io.parsers import read_csv
from tqdm import tqdm
from multiprocessing import Pool
from mlutils import imread, Log

from .utils import get_roi, Resize2D



def roi_crop(image, resize):
    num_labels, roi_list = get_roi(image)
    x, y, w, h = roi_list[0]
    image = image[y:y + h, x:x + w, :]
    image = resize(image)
    return image


def save_pil(gen_path, jpg_path, image):
    pil_path = jpg_path_to_pil_path(gen_path, jpg_path)
    with open(pil_path, 'wb') as f:
        pickle.dump(image, f)

def jpg_path_to_pil_path(gen_path, jpg_path):
    jpg_name = jpg_path.split('/')[-1]
    image_id = jpg_name.split('.')[0]
    pil_path = os.path.join(gen_path, f'{image_id}.PIL')
    return pil_path


def run(param):
    image_path = param[0]
    original_path = param[1]
    gen_path = param[2]
    resize = param[3]

    image = imread(image_path)
    try:
        image = roi_crop(image, resize)
    except:
        Log.info(image_path)
        image = resize(image)
    image = Image.fromarray(np.uint8(image))
    save_pil(gen_path, image_path, image)


class Generator(object):
    SUFFIX = 'jpeg'

    def __init__(self, opt):
        if opt.gen_test is True:
            self.original_path = opt.test_original_path
            self.gen_path = opt.test_gen_path
        else:
            self.original_path = opt.train_original_path
            self.gen_path = opt.train_gen_path

        self.ps = opt.ps
        self.file_list = []
        glob_path = os.path.join(self.original_path, f'*.{self.SUFFIX}')
        for file_path in glob.glob(glob_path):
            self.file_list.append(file_path)
        self.file_cnt = len(self.file_list)
        self.resize = Resize2D(opt.size)
        self.rm_old_file()

    def rm_old_file(self):
        Log.info('Remove old generated files.')
        if os.path.exists(self.gen_path):
            shutil.rmtree(self.gen_path)
        os.mkdir(self.gen_path)

    def generate(self):
        Log.info(f'Convert: {self.file_cnt}')
        task_list = []
        for file_path in self.file_list:
            task_list.append((file_path, self.original_path,
                            self.gen_path, self.resize))

        with Pool(self.ps) as p:
            _ = list(tqdm(p.imap(run, task_list), total=self.file_cnt))


class MetaDataGen(object):
    def __init__(self, opt):
        super().__init__()
        self.anno_path = opt.annotation_path
        self.gen_path = opt.train_gen_path

    @staticmethod
    def read_csv(path):
        data = pd.read_csv(path, sep=',')
        return data

    def name_to_path(self, name):
        pil_name = f'{name}.PIL'
        return os.path.join(self.gen_path, pil_name)

    def parse(self):
        df = read_csv(self.anno_path)
        output_data = []
        for i in range(len(df)):
            image_name = df.iloc[i]['image']
            label = df.iloc[i]['level']
            image_path = self.name_to_path(image_name)
            output_data.append([image_path, label])
        return output_data


def gen_train(opt):
    train_generator = Generator(opt)
    train_generator.generate()

    anno_paser = MetaDataGen(opt)
    meta_data = anno_paser.parse()

    meta_data_path = os.path.join(opt.train_gen_path, 'meta.pickle')
    with open(meta_data_path, 'wb') as f:
        pickle.dump(meta_data, f)


def gen_test(opt):
    train_generator = Generator(opt)
    train_generator.generate()
