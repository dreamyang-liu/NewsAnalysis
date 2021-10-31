import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from Define import *

import re
import gc
import demjson
import logging
import os
import copy
from flask import Flask, request
from flask_cors import *
from SystemModules import *
import json

class SystemProxy(object):

    def __init__(self):
        self.modules = {
            SystemModuleType.TS: TSSystemModule('t5'),
            SystemModuleType.FD: FDSystemModule('att'),
        }

    def get_module(self, module_type:SystemModuleType):
        return self.modules.get(module_type)
    
    def request_dispatch(self, request):
        # Maybe use a model to decide dispatch ???
        if request['task_type'] == 'ts':
            module_type = SystemModuleType.TS
        elif request['task_type'] == 'fd':
            module_type = SystemModuleType.FD
        module = self.get_module(module_type)
        return module.handle(request)
    
    def serve(self, request):
        return self.request_dispatch(request)


PROXY = SystemProxy()

def model_handler(kwargs):
    request = kwargs
    model_prediction_dict = PROXY.serve(request)
    gc.collect()
    res = dict()
    res['fd'] = model_prediction_dict['fd']
    res['summary'] = model_prediction_dict['summary']
    res['answer'] = model_prediction_dict['answer']
    return res

app = Flask(__name__)

CORS(app, supports_credentials=True)

@app.route('/api/model/predict', methods=['POST'])
def test():
    gc.collect()
    if request.method == 'POST':
        data = request.get_data(as_text=False)
        data_dict = json.loads(data)['data']
        # print(data_dict)
        response = dict()
        response['status'] = 200
        try:
            response['data'] = model_handler(data_dict)
            gc.collect()
        except Exception as e:
            logging.exception('Exception occured while handling data')
            response['status'] = 503
        return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000")