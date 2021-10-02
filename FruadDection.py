import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm, trange
from argparser import args, DEVICE
from typing import Optional, Union
from transformers import AutoTokenizer

OPT_DICT = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
}

LOSS_DICT = {
    'ce': nn.CrossEntropyLoss()
}

class EmbeddingLayer(nn.Module):

    def __init__(self, embed_dim, vocab_dim):
        super(EmbeddingLayer, self).__init__()
        self._embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=0)
    
    def forward(self, inputs):
        pass

class ClassifierLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ClassifierLayer, self).__init__()

class FruadDection(nn.Module):
    
    def __init__(self):
        super(FruadDection, self).__init__()
        self._word_embedding = Union[nn.Module, torch.Tensor, None]
        self._classifier_layers = Union[nn.Module, None]

    def forward(self, X):
        embedding = self._word_embedding.forward(X)
        score = F.softmax(self._classifier_layers.forward(embedding))
        return score

class FraudDectionTrainer(object):

    def __init__(self, model:nn.Module, train_feature:torch.Tensor, train_label:torch.Tensor, test_feature:torch.Tensor, test_label:torch.Tensor, args):
        self.model = model.to(DEVICE)
        self.train_feature = train_feature
        self.train_label = train_label
        self.test_feature = test_feature
        self.test_label = test_label
        self.args = args

        self.batch_size = self.args.batch_size
        self.loss_type = self.args.loss_type
        self.epoch = self.args.epoch
        self.learning_rate = self.args.learning_rate
        self.optimizer_type = self.args.optimizer_type

        self.optim = OPT_DICT[self.optimizer_type](self.model.parameters(), lr=self.learning_rate)
        self.loss = LOSS_DICT[self.loss_type]
        
    def train(self):
        self.model.train()
        with trange(self.epoch) as progress:
            for ep in progress:
                total_loss = 0
                for batch_idx in range(0, (self.train_feature.shape[0] // self.batch_size) + 1):
                    start_idx = min((batch_idx + 1) * self.batch_size, self.train_feature.shape[0])
                    end_idx = min((batch_idx + 1) * self.batch_size, self.train_feature.shape[0])

                    batch = self.train_feature[start_idx:end_idx].to(DEVICE)
                    scores = self.model.forward(batch)
                    batch_label = self.train_label[start_idx:end_idx].to(DEVICE)
                    loss = self.loss(scores, batch_label)
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    total_loss += loss.item()
                progress.set_description(f"epoch{ep} loss: {total_loss}")

    def eval(self, X):
        self.model.eval()
        total_loss = 0
        for batch_idx in range(0, (self.test_feature.shape[0] // self.batch_size) + 1):
            start_idx = min((batch_idx + 1) * self.batch_size, self.test_feature.shape[0])
            end_idx = min((batch_idx + 1) * self.batch_size, self.test_feature.shape[0])

            batch = self.test_feature[start_idx:end_idx].to(DEVICE)
            scores = self.model.forward(batch)
            batch_label = self.test_label[start_idx:end_idx].to(DEVICE)
            loss = self.loss(scores, batch_label)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            total_loss += loss.item()
        print(f"Evaluation complete, total loss: {total_loss}")


