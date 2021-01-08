import glob
import os
import shutil
import re
import pickle
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from mlutils import imread, Log
from multiprocessing import Pool

from .utils import get_roi, Resize2D


class DataParser(object):
    ID = 'ID'
    MALE = 'Male'
    FEMALE = 'Female'
    LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    LEFT_DIAGN = 'Left-Diagnostic Keywords'
    RIGHT_DIAGN = 'Right-Diagnostic Keywords'

    LEFT_FUNDUS = 'Left-Fundus'
    RIGHT_FUNDUS = 'Right-Fundus'
    AGE = 'Patient Age'
    SEX = 'Patient Sex'

    EXCLUSIVE = ['2174', '2175', '2176', '2177',
                 '2178', '2179', '2180', '2181',
                 '2182', '2957']

    def __init__(self, opt):
        self.enabel_exclusive = opt.enabel_exclusive
        self.gen_path = opt.train_gen_path
        self.annotations = pd.read_excel(opt.annotation_path)

    def parse_diagnosis(self, keywords):
        result = {'N': 0, 'D': 0, 'G': 0, 'C': 0, 'A': 0,
                  'H': 0, 'M': 0, 'O': 0}
        
        items = re.split('，|,', keywords)
        normal_flag = True
        for item in items:
            if 'proliferative retinopathy' in item or\
            'diabetic retinopathy' in item:
                result['D'] = 1
                normal_flag = False
                continue
            elif 'myopi' in item:
                result['M'] = 1
                normal_flag = False
                continue
            elif 'age-related macular degeneration' in item:
                result['A'] = 1
                normal_flag = False
                continue
            elif 'glaucoma' in item:
                result['G'] = 1
                normal_flag = False
                continue
            elif 'cataract' in item:
                result['C'] = 1
                normal_flag = False
                continue
            elif 'hypertensive retinopathy' in item:
                result['H'] = 1
                normal_flag = False
                continue
            elif 'lens dust' in item or\
                'optic disk photographically invisible' in item or\
                'image' in item:
                continue
            elif 'normal fundus' != item:
                result['O'] = 1
                normal_flag = False

        if normal_flag:
            result['N'] = 1
        return result

    def assert_parser(self, anno_item, parsed_right, parsed_left):
        origin = anno_item[self.LABELS]
        result = True
        for label in self.LABELS:
            if label != 'N':
                if origin[label] != (parsed_right[label] | parsed_left[label]):
                    result = False
            else:
                if origin[label] != (parsed_right[label] & parsed_left[label]):
                    result = False

        if not result:
            Log.error('=================[error]==================')
            Log.error(anno_item)
            Log.error('right', parsed_right)
            Log.error('left', parsed_left)
            Log.error(self.combo_label(anno_item))
            assert(False)

    def dict_to_label(self, dict_label):
        lst = []
        for i, label in enumerate(self.LABELS):
            if dict_label[label] == 1:
                lst.append(i)
        return np.array(lst)

    def combo_label(self, anno_item):
        lst = []
        for i, label in enumerate(self.LABELS):
            if anno_item[label] == 1:
                lst.append(i)
        return np.array(lst)
    
    def exclusive(self, anno_item):
        if self.enabel_exclusive:
            anno_id = str(anno_item[self.ID])
            if anno_id in self.EXCLUSIVE:
                return True
        return False

    @classmethod
    def translate_sex(cls, sex):
        if sex == cls.MALE:
            return 0.0
        elif sex == cls.FEMALE:
            return 1.0
        else:
            raise RuntimeError(f'Unrecognized sex ({sex})')

    @staticmethod
    def translate_age(age):
        return float(age)

    def jpg_to_pil_path(self, jpg_name):
        name = jpg_name.split('.')[0]
        return os.path.join(self.gen_path, f'{name}.PIL')

    def parse(self):
        Log.info('Prasing annotations')
        size = self.annotations.shape[0]

        meta_data = []
        for i in range(size):
            anno_item = self.annotations.iloc[i]
            if self.exclusive(anno_item):
                continue
            
            right_diagn = anno_item[self.RIGHT_DIAGN]
            left_diagn = anno_item[self.LEFT_DIAGN]
            left_eye = anno_item[self.LEFT_FUNDUS]
            right_eye = anno_item[self.RIGHT_FUNDUS]
            age = anno_item[self.AGE]
            sex = anno_item[self.SEX]

            dict_right_labels = self.parse_diagnosis(right_diagn)
            dict_left_labels = self.parse_diagnosis(left_diagn)
            self.assert_parser(anno_item, dict_right_labels, dict_left_labels)

            meta_data.append(
                {'age': self.translate_age(age),
                 'sex': self.translate_sex(sex),
                 'left_eye': self.jpg_to_pil_path(left_eye),
                 'right_eye': self.jpg_to_pil_path(right_eye),
                 'left_label': self.dict_to_label(dict_left_labels),
                 'right_label': self.dict_to_label(dict_right_labels),
                 'combo_label': self.combo_label(anno_item)}
            )
        return meta_data


class Generator(object):
    def __init__(self, opt):
        if opt.gen_test is True:
            self.original_path = opt.test_original_path
            self.gen_path = opt.test_gen_path
        else:
            self.original_path = opt.train_original_path
            self.gen_path = opt.train_gen_path

        self.file_list = []
        for file_path in glob.glob(os.path.join(self.original_path, '*.jpg')):
            self.file_list.append(file_path)
        self.file_cnt = len(self.file_list)
        self.resize = Resize2D(opt.size)
        self.rm_old_file()
        self.ps = opt.ps

    def rm_old_file(self):
        Log.info('Remove old generated files.')
        if os.path.exists(self.gen_path):
            shutil.rmtree(self.gen_path)
        os.mkdir(self.gen_path)

    def imread(self, path):
        return imread(path)

    def jpg_path_to_pil_path(self, jpg_path):
        jpg_name = jpg_path.split('/')[-1]
        image_id = jpg_name.split('.')[0]
        pil_path = os.path.join(self.gen_path, f'{image_id}.PIL')
        return pil_path

    def roi_crop(self, image):
        num_labels, roi_list = get_roi(image)
        # assert num_labels == 1

        x, y, w, h = roi_list[0]
        image = image[y:y + h, x:x + w, :]
        image = self.resize(image)
        return image

    def save_pil(self, jpg_name, image):
        pil_name = self.jpg_path_to_pil_path(jpg_name)
        with open(pil_name, 'wb') as f:
            pickle.dump(image, f)

    def run(self, file_path):
        image = self.imread(file_path)
        image = self.roi_crop(image)
        image = Image.fromarray(np.uint8(image))
        self.save_pil(file_path, image)

    def generate(self):
        task_list = []
        for file_path in self.file_list:
            task_list.append((file_path))

        with Pool(self.ps) as p:
            _ = list(tqdm(p.imap(self.run, task_list), total=self.file_cnt))


def gen_train(opt):
    train_generator = Generator(opt)
    train_generator.generate()

    anno_paser = DataParser(opt)
    meta_data = anno_paser.parse()

    meta_data_path = os.path.join(opt.train_gen_path, 'meta.pickle')
    with open(meta_data_path, 'wb') as f:
        pickle.dump(meta_data, f)


def gen_test(opt):
    train_generator = Generator(opt)
    train_generator.generate()
