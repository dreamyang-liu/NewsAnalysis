from abc import abstractmethod
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import re

from typing import Union, Optional

class DataProcesser(object):

    def __init__(self, *args, **kwargs):
        self.data = Union[pd.DataFrame, str, None]
        self.raw = Union[pd.DataFrame, str, None] # Used for original data archive
    
    @staticmethod
    def _read_csv(filename):
        return pd.read_csv(filename)
    
    @staticmethod
    def _read_txt(filename):
        with open(filename, 'r') as f:
            return f.read()
    
    def read(self, filename:str):
        if filename.endswith(".csv"):
            self.data = self._read_csv(filename)
            self.raw = self._read_csv(filename)
        else:
            self.data = self._read_txt(filename)
            self.raw = self._read_txt(filename)
    
    @staticmethod
    def split_feature_label():
        raise NotImplementedError("split_feature_label not implemented in subclass")

class FakeNewsDataProcesser(DataProcesser):

    @staticmethod
    def split_feature_label(data):
        label = data.label
        feature = data.drop(['label'], axis=1)
        return feature, label
    
    def imputation(self):
        self.data.author.fillna('Anonymous', inplace=True)
    
    @staticmethod
    def check_author(data):
        if data == 'nan':
            return 'Anonymous'
        return data if len(data) < 50 else 'Author Sentence'

    def remove_invalid_char(self):
        self.data.text = self.data.text.apply(lambda x: x.replace('@', ' at '))
        self.data.text = self.data.text.apply(lambda x: x if len(x) > 10 else np.nan)
        self.data.text = self.data.text.apply(lambda x: re.sub('&|\*|#', '', str(x)))
        self.data.text = self.data.text.apply(lambda x: x.replace(u'\u2026', ''))
        self.data.text = self.data.text.apply(lambda x: x.replace('|', ''))
        self.data.text = self.data.text.apply(lambda x: x.lower())
        self.data.text = self.data.text.apply(lambda x: x.replace('-', ' '))
        self.data.text = self.data.text.apply(lambda x: x.replace(' – ', ' '))
        self.data.title = self.data.title.apply(lambda x: x if len(x) > 5 else np.nan)
        self.data.author = self.data.author.apply(lambda x: self.check_author(x))
        self.data.author = self.data.author.apply(lambda x: re.sub(r'^\s+&', 'Anonymous', x))
        self.data.author = self.data.author.apply(lambda x: re.split('and|,', x))
    
    def remove_nan_text(self):
        self.data = self.data[self.data.text.notna()]
        self.data = self.data[self.data.title.notna()]
    
    def default_process(self, split_dataset=False, vali_set=False, split_ratio:float=0.8, eval=False):
        if eval:
            self.remove_nan_text()
            self.data.label = 1
            self.imputation()
            self.remove_invalid_char()
            self.remove_nan_text()
            return self.data
        

        self.remove_nan_text()
        self.imputation()
        self.remove_invalid_char()
        self.remove_nan_text()
        feature = dict()
        label = dict()
        if split_dataset:
            if vali_set:
                vali_start = int(self.data.shape[0] * split_ratio)
                test_start = int(self.data.shape[0] * (split_ratio + .1))
                train_data = self.data[0:vali_start]
                vali_data = self.data[vali_start:test_start]
                test_data = self.data[test_start:]
                feature['vali'], label['vali'] = self.split_feature_label(vali_data)
            else:
                test_start = int(self.data.shape[0] * split_ratio)
                train_data, test_data = self.data[0:test_start], self.data[test_start:]
            feature['train'], label['train'] = self.split_feature_label(train_data)
            feature['test'], label['test'] = self.split_feature_label(test_data)
        return feature, label
            

if __name__ == "__main__":
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)

    feature, label = dp.default_process(split_dataset=True)
    np.set_printoptions(threshold=sys.maxsize)