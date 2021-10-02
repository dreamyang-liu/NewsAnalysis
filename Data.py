import pandas as pd
import numpy as np
import pickle as pkl
class DataProcesser(object):

    def __init__(self, *args, **kwargs):
        self.data = None
    
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
        else:
            self.data = self._read_txt(filename)

if __name__ == "__main__":
    file = './data/iclr2017_conversations.csv'
    dp = DataProcesser()
    dp.read(file)
    dp.data.info()