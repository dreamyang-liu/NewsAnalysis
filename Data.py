from abc import abstractmethod
import pandas as pd
import numpy as np
import pickle as pkl

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
        self.data.author.fillna('Unknown', inplace=True)
        self.data.text.fillna('No Content', inplace=True)
        self.data.title.fillna('No Title', inplace=True)
    
    def remove_invalid_char(self):
        self.data.text = self.data.text.apply(lambda x: x.replace('@', ' at '))
        self.data.text = self.data.text.apply(lambda x: x.replace(r"[^A-Za-z0-9,.!'?&]", " "))
    
    def default_process(self, split_train_test=False, split_ratio=0.8):
        self.imputation()
        self.remove_invalid_char()
        if split_train_test:
            test_start = int(self.data.shape[0] * split_ratio)
            train_data, test_data = self.data[0:test_start], self.data[test_start:]
            return self.split_feature_label(train_data), self.split_feature_label(test_data)
        return self.split_feature_label(self.data)
            

    

if __name__ == "__main__":
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)

    (train_feature, train_label), (test_feature, test_label) = dp.default_process(split_train_test=True)