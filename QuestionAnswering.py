from transformers import DistilBertForQuestionAnswering
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction
from transformers import BertTokenizer, BertForQuestionAnswering, BertTokenizerFast, BertForTokenClassification
import torch
from datasets import load_dataset
import numpy
from transformers import AutoModelForQuestionAnswering
from transformers import DistilBertTokenizerFast
from transformers import pipeline

"""
class QuestionAnsweringTrainer(object):

    def __init__(self):
        self.choice = "distilbert"
        self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def preprocess(self):

    def train(self):


    def evaluate(self):

"""

class QuestionAnsweringModel(object):
    def __init__(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained('aszidon/' + 'distilbert' + 'custom5')
        self.tokenizer = AutoTokenizer.from_pretrained('aszidon/' + 'distilbert' + 'custom5')
        self.pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="pt")

    def getanswer(quest, cont):
        return pipe(question=quest, context=cont)['answer']
