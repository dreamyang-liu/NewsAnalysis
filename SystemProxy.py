import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import math
from Define import *

import re
import gc
#import demjson
import logging
import os
import copy
from flask import Flask, request, render_template, url_for,jsonify
from flask_cors import *
from SystemModules import *
import json

class SystemProxy(object):

    def __init__(self):
        pass
        #self.modules = {
        #    SystemModuleType.TS: TSSystemModule('t5'),
        #    SystemModuleType.FD: FDSystemModule('att'),
        #    SystemModuleType.QA: QASystemModule('distill-bert'),
        #}
    '''
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
    '''

    def serve(self, request_c):
        #return self.request_dispatch(request)
        mod_sum=TSSystemModule('t5')
        mod_sum.initialize()
        mod_fd=FDSystemModule('att')
        mod_fd.initialize()
        mod_qa=QASystemModule('distill_bert')
        mod_qa.initialize()
        res={}
        res['summary']=mod_sum.handle(request_c.get('text'))
        res['fruad']=mod_fd.handle(request_c)
        res['answer']=mod_qa.handle(request_c)
        res['question']=request_c.get('question')
        return res


'''
def model_handler(kwargs):
    request = kwargs
    model_prediction_dict = PROXY.serve(request)
    gc.collect()
    res = dict()
    res['fd'] = model_prediction_dict['fd']
    res['summary'] = model_prediction_dict['summary']
    res['answer'] = model_prediction_dict['answer']
    return res
'''

PROXY = SystemProxy()
app = Flask(__name__)
CORS(app, supports_credentials=True)

#@app.route('/api/model/predict', methods=['POST'])

@app.route('/api', methods=['POST'])
def index():
    gc.collect()
    if request.method == 'POST':
        data = request.get_data(as_text=False)
        data_dict = json.loads(data)
        response = {}
        response['status'] = 200
        try:
            #response['data'] = model_handler(data_dict)
            response['data']=PROXY.serve(data_dict)
            gc.collect()
        except Exception as e:
            logging.exception('Exception occured while handling data')
            response['status'] = 503
        print(response)
        return response

        #return render_template('index.html', QTc_result=(1,True))
'''




@app.route('/', methods=['GET', 'POST'])
def index():
  QTc_result = False
  if request.method == 'POST':
    form = request.form
    QTc_result = calculate_qtc(form)
  return render_template('index.html', QTc_result=QTc_result)
def calculate_qtc(form):
  sex = request.form['sex']
  heart_rate = int(request.form['hr'])
  qt_int = int(request.form['qt'])

  qt_seconds = qt_int / 1000
  rr_interval = (6000 / heart_rate)
  QTc = qt_seconds / math.sqrt(rr_interval)
  formated_QTc = round((QTc * 1000) * 10, 0)
  if (formated_QTc > 440 and sex == 'm') or (formated_QTc > 460 and sex == 'f'):
    prolonged = True
  else:
    prolonged = False
  return (formated_QTc, prolonged)

'''
if __name__ == '__main__':
    app.run(debug=True)
