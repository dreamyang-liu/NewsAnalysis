import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo as torchinfo
import pysbd
import pdb

from utils import *
from bidict import bidict, inverted
from tqdm import tqdm, trange
from argparser import args, DEVICE, CPU
from Datagenerator import DataGenerator
from Data import FakeNewsDataProcesser
from FDModel import *

DEBUG = False


class FraudDectionTrainer(object):

    def __init__(self, model, data_generator:DataGenerator, args):
        self.model = model.to(TRAIN_DEVICE)
        self.data_generator = data_generator
        self.author_embedding = self.data_generator.author_embedding

        self.args = args

        self.batch_size = self.args.batch_size
        self.loss_type = self.args.loss_type
        self.epoch = self.args.epoch
        self.learning_rate = self.args.learning_rate

        self.optim = optim.Adam(list(self.model.parameters()) + list(self.author_embedding.parameters()), lr=0.01)
        self.loss = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        with trange(self.epoch) as progress:
            author_emb, title_emb, text_emb, label = self.data_generator.get_train_features()
            label = label.to(TRAIN_DEVICE)
            for ep in progress:
                try:
                    o = self.model(author_emb, title_emb, text_emb)
                    loss = self.loss(o, label)
                    out = torch.argmax(o.detach(), dim=1)
                    acc = (out.shape[0] - torch.count_nonzero(torch.logical_xor(out, label.detach()))) / out.shape[0]
                    loss.backward(retain_graph=True)
                    self.optim.step()
                    self.optim.zero_grad()
                    progress.set_description(f"epoch{ep} train loss: {loss}, acc: {acc}")
                except StopIteration:
                    pass
        torch.save(self.model.state_dict(), self.args.save_dir)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        vali_loss = 0
        
        return total_loss, vali_loss

    def eval_epoch(self):
        self.model.eval()
        pass
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir))
    
    def eval_on_vali(self):
        self.load_model()
        self.model.eval()
        total_loss = 0
        for batch_idx in range(0, (self.vali_feature.shape[0] // self.batch_size) + 1):
            start_idx = min((batch_idx + 1) * self.batch_size, self.vali_feature.shape[0])
            end_idx = min((batch_idx + 1) * self.batch_size, self.vali_feature.shape[0])

            batch = self.vali_feature[start_idx:end_idx].to(DEVICE)
            scores = self.model.forward(batch)
            batch_label = self.vali_label[start_idx:end_idx].to(DEVICE)
            loss = self.loss(scores, batch_label)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            total_loss += loss.item()
        print(f"Evaluation complete, total loss: {total_loss}")

    def eval_on_test(self):
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


if __name__ == "__main__":
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)
    feature, label = dp.default_process(split_dataset=True, vali_set=True)
    data_generator = DataGenerator(feature, label)
    model = FDModel(32, 768)
    trainer = FraudDectionTrainer(model, data_generator, args)
    trainer.train()

    # se = SentenceEmbedding()
    # se.get_sentence_embeddings(None)
