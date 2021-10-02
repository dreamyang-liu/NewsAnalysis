import torch

from abc import ABC, abstractmethod
from Define import *

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class SystemModuleBase(ABC):

    def __init__(self, type:SystemModuleType, tag:str):
        self.type = type
        self.tag = tag

    @abstractmethod
    def handle(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

class TSSystemModule(SystemModuleBase):

    def __init__(self, tag:str, *args, **kwargs):
        self.tag = tag
        super(TSSystemModule, self).__init__(SystemModuleType.TS, tag, *args, **kwargs)

    def initialize(self):
        
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    
    def handle(self, request):
        ARTICLE_TO_SUMMARIZE = request
        inputs = self.tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]


class QASystemModule(SystemModuleBase):

    def __init__(self, tag:str, *args, **kwargs):
        self.tag = tag
        super(QASystemModule, self).__init__(SystemModuleType.QA, tag, *args, **kwargs)

    def initialize(self):
        model_name = "deepset/roberta-base-squad2"
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def digest_context(self, passage):
        return passage
    
    def handle(self, request):

        question = request
        context = self.digest_context()
        encoding = self.tokenizer.encode_plus(question, context, return_tensors="pt", max_length=4096)
        input_ids = encoding["input_ids"].to("cuda")
        attention_mask = encoding["attention_mask"].to("cuda")

        # the forward method will automatically set global attention on question tokens
        # The scores for the possible start token and end token of the answer are retrived
        # wrap the function in torch.no_grad() to save memory
        with torch.no_grad():
            start_scores, end_scores = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Let's take the most likely token using `argmax` and retrieve the answer
        all_tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
        answer_tokens = all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1]
        answer = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(answer_tokens))[1:].replace('"', '')
        return answer

class FDSystemModule(SystemModuleBase):

    def __init__(self, tag:str, *args, **kwargs):
        self.tag = tag
        super(FDSystemModule, self).__init__(SystemModuleType.FD, tag, *args, **kwargs)
    
    def initialize(self):
        pass
    
    def handle(self, request):
        pass


class SASystemModule(SystemModuleBase):

    def __init__(self, tag:str, *args, **kwargs):
        self.tag = tag
        super(SASystemModule, self).__init__(SystemModuleType.SA, tag, *args, **kwargs)
    
    def initialize(self):
        pass

    def handle(self, request):
        pass


if __name__ == "__main__":
    qa = QASystemModule(tag=1)
    qa.initialize()