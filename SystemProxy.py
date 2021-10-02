import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from Define import *

class SystemProxy(object):

    def __init__(self, modules:dict):
        self.modules = modules

    def get_module(self, module_type:SystemModuleType):
        return self.modules.get(module_type)
    
    def request_dispatch(self, request):
        # Maybe use a model to decide dispatch ???
        module_type = SystemModuleType.QA
        module = self.get_module(module_type)
        return module.handle(request)
    
    def serve(self, request):
        return self.request_dispatch(request)