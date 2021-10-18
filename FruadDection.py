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

PRETRAIN_DEVICE = torch.device("cuda:2")
TRAIN_DEVICE = torch.device("cuda:1")
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

        self.optim = optim.SGD(list(model.parameters()) + list(self.author_embedding.parameters()), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        with trange(self.epoch) as progress:
            author_emb, title_emb, text_emb, label = self.data_generator.get_train_features()
            for ep in progress:
                try:
                    print(author_emb[0])
                    print(title_emb[0])
                    print(text_emb[0])
                    print(label[0])
                except StopIteration:
                    pass
                # progress.set_description(f"epoch{ep} train loss: {total_loss}, vali loss: {vali_loss}")
                # torch.save(self.model.state_dict(), self.args.save_dir)
    
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
