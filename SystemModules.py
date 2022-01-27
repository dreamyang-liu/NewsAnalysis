import torch

from abc import ABC, abstractmethod
from Define import *

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, AutoModelForSeq2SeqLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from Datagenerator import *
from FruadDection import *

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

        self.tokenizer = AutoTokenizer.from_pretrained("bochaowei/t5-small-finetuned-cnn-wei1")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("bochaowei/t5-small-finetuned-cnn-wei1")

    def handle(self, request):
        prefix='summarize: '
        ARTICLE_TO_SUMMARIZE=prefix
        for doc in request:
            if doc not in '!#$%&\()*+,-./:;<=>?@[\\]^_{|}~`':
               ARTICLE_TO_SUMMARIZE+=doc
        input_ids = self.tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors='pt').input_ids
        outputs = self.model.generate(input_ids,max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


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
        self.interface = FraudDectionInterface()
    
    def handle(self, request):
        author = request.get('author')
        title = request.get('title')
        text = request.get('text')
        score = self.interface.predict(author, title, text)
        return score

class SASystemModule(SystemModuleBase):

    def __init__(self, tag:str, *args, **kwargs):
        self.tag = tag
        super(SASystemModule, self).__init__(SystemModuleType.SA, tag, *args, **kwargs)

    def initialize(self):
        pass



if __name__ == "__main__":
    qa = QASystemModule(tag=1)
    qa.initialize()

