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
from typing import Optional, Union
from transformers import  AutoTokenizer, AutoModel
from Data import FakeNewsDataProcesser

PRETRAIN_DEVICE = torch.device("cuda:2")
TRAIN_DEVICE = torch.device("cuda:1")
DEBUG = False

class AuthorEmbedding(nn.Module):

    def __init__(self, author_dim, embed_dim):
        super(AuthorEmbedding, self).__init__()
        self._embedding = nn.Embedding(author_dim, embed_dim)
    
    def forward(self, inputs):
        return self._embedding(inputs)

class AuthorTokenizer(object):
    
    def __init__(self, author_list):
        self.map = bidict({author:i+1 for i, author in enumerate(set(author_list))})
        self.map['###'] = 0
    
    def encode_author(self, authors):
        if isinstance(authors, str):
            return torch.tensor([self.map[authors] if authors in dict(self.map).keys() else 0], dtype=torch.int16)
        if isinstance(authors, list):
            return torch.tensor([self.map[x] if x in dict(self.map).keys() else 0 for x in authors], dtype=torch.int16)


class TSAttention(nn.Module):
    
    def __init__(self, title_dim, sentence_dim):
        super(TSAttention, self).__init__()
        self.fc1 = nn.Linear(sentence_dim, title_dim, bias=False)
        self.atac = nn.Tanh()
        self.out = nn.Softmax()
    
    def forward(self, title_emb:torch.Tensor, sentence_emb:torch.Tensor):
        attn_score = F.softmax(title_emb * self.atac(self.fc1(sentence_emb)))
        attn_emb = attn_score @ sentence_emb
        return title_emb, attn_emb

class ASAttention(nn.Module):
    def __init__(self, author_dim, sentence_dim):
        super(ASAttention, self).__init__()
        self.fc1 = nn.Linear(sentence_dim, author_dim, bias=False)
        self.atac = nn.Tanh()
        self.out = nn.Softmax()
    
    def forward(self, author_emb:torch.Tensor, title_emb:torch.Tensor, sentence_emb:torch.Tensor):
        ts_emb = torch.cat([title_emb, sentence_emb], dim=0)
        attn_emb_score = F.softmax(author_emb * self.atac(self.fc1(ts_emb)))
        print(f'attn_emb_score shape: {attn_emb_score.shape}')
        attn_emb = attn_emb_score @ ts_emb
        return author_emb, attn_emb


class ClassifierLayer(nn.Module):

    def __init__(self, author_embed_dim, sentence_embed_dim, short_cut=False):
        super(ClassifierLayer, self).__init__()
        if short_cut:
            self.fc1 = nn.Linear((author_embed_dim + sentence_embed_dim * 3), 2)
        else:
            self.fc1 = nn.Linear((author_embed_dim + sentence_embed_dim), 2)
        self.out = nn.Softmax()
    
    def forward(self, author_embedding, sententce_embeddings:torch.Tensor):
        return self.out(self.fc1(torch.cat([author_embedding, sententce_embeddings.flatten()], dim=1)))
        

class FruadDection(nn.Module):
    
    def __init__(self, author_len, author_embed_dim, sentence_embed_dim):
        super(FruadDection, self).__init__()
        self._author_embedding = AuthorEmbedding(author_len, author_embed_dim)
        self._classifier_layers = ClassifierLayer(author_embed_dim, sentence_embed_dim)
        self.ts_attention = TSAttention(sentence_embed_dim, sentence_embed_dim)
        self.as_attention = ASAttention(author_embed_dim, sentence_embed_dim)

    def forward(self, author_ids, title_embedding, sententce_embedding):
        author_embed = self._author_embedding(author_ids)
        title_emb, sentence_attn_embed = self.ts_attention.forward(title_embedding, sententce_embedding)
        authro_emb, ts_attn = self.as_attention.forward(author_embed, title_emb, sentence_attn_embed)
        score = self._classifier_layers.forward(authro_emb, ts_attn)
        return score


class SentenceEmbedding(object):

    def __init__(self):

        #Mean Pooling - Take attention mask into account for correct averaging

        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2').to(PRETRAIN_DEVICE)
        freeze_model(self.model)

        # Tokenize sentences
        # Perform pooling. In this case, max pooling.
    
    def get_sentence_embeddings(self, sentences):
        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(PRETRAIN_DEVICE)
        # Compute token embeddings
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.to(CPU)
        encoded_input = encoded_input.to(CPU)
        return sentence_embeddings
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class FraudDectionTrainer(object):

    def __init__(self, feature:dict, label:dict, args):

        self._sentence_embeddings = SentenceEmbedding()

        if 'vali' in feature.keys():
            self.vali_feature = feature['vali']
            self.vali_label = label['vali']

        self.train_feature = feature['train']
        self.train_label = label['train']
        
        self.test_feature = feature['test']
        self.test_label = label['test']

        self.args = args

        self.batch_size = self.args.batch_size
        self.loss_type = self.args.loss_type
        self.epoch = self.args.epoch
        self.learning_rate = self.args.learning_rate

        
        self.loss = nn.CrossEntropyLoss()
    
    def initialize(self, author_embed_dim, sentence_embed_dim):
        author_len = self._author_tokenizer.map.__len__()
        self.model = FruadDection(author_len, author_embed_dim, sentence_embed_dim)
        self.optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
    def convert_text_to_tensor(self, feature, tag, cache=True):
        # print(feature['text'], len(feature['text']))
        if cache:
            try:
                authors = torch.load(f'saved_models/{tag}-authors')
                title_embeddings = torch.load(f'saved_models/{tag}-title_embeddings')
                text_sentences_embedding = torch.load(f'saved_models/{tag}-text_sentences_embedding')
                return authors, title_embeddings, text_sentences_embedding
            except Exception:
                pass
        segmenter = pysbd.Segmenter()
        titles = []
        for title in feature['title']:
            if title != ' ':
                titles.append(title)
            else:
                titles.append('###')
        
        text_sentences = []

        for text in tqdm(feature['text']):
            sentences = segmenter.segment(text)
            if len(sentences) > 0:
                text_sentences.append(sentences)
            else:
                text_sentences.append(['###'])
        
        authors = []
        for author in feature['author']:
            if author != ' ':
                authors.append(author)
            else:
                authors.append('###')
        
        author_tokenizer = AuthorTokenizer(authors)
        authors = author_tokenizer.encode_author(authors)

        self._author_tokenizer = author_tokenizer
        title_embeddings = torch.tensor(torch.rand(len(titles), 768))
        title_batch = 2000
        for batch_idx in range(len(titles) // title_batch + 1):
            batch_start = batch_idx * title_batch
            batch_end = min(batch_start + title_batch, len(titles))
            batch_titles = titles[batch_start:batch_end]
            title_embeddings[batch_start:batch_end, ...] = self._sentence_embeddings.get_sentence_embeddings(batch_titles)

        gpu_memory_collect()

        text_sentences_embedding = []
        for ipt in text_sentences:
            out = self._sentence_embeddings.get_sentence_embeddings(ipt)
            text_sentences_embedding.append(out)
            gpu_memory_collect()

        torch.save(authors, f'saved_models/{tag}-authors')
        torch.save(title_embeddings, f'saved_models/{tag}-title_embeddings')
        torch.save(text_sentences_embedding, f'saved_models/{tag}-text_sentences_embedding')
        return authors, title_embeddings, text_sentences_embedding

        
    def train(self):
        self.model.train()
        with trange(self.epoch) as progress:
            for ep in progress:
                total_loss = 0
                for batch_idx in range(0, (self.train_feature.shape[0] // self.batch_size) + 1):
                    start_idx = min((batch_idx + 1) * self.batch_size, self.train_feature.shape[0])
                    end_idx = min((batch_idx + 1) * self.batch_size, self.train_feature.shape[0])
                    batch = self.train_feature[start_idx:end_idx]
                    batch_author_ids, batch_title_embed, batch_sentences_embed = self.convert_text_to_tensor(batch, 'train')
                    batch_author_ids = batch.author_ids.to(DEVICE)
                    batch_title_embed = batch_title_embed.to(DEVICE)
                    batch_sentences_embed = batch_sentences_embed.to(DEVICE)
                    scores = self.model.forward(batch_author_ids, batch_title_embed, batch_sentences_embed)
                    batch_label = self.train_label[start_idx:end_idx].to(DEVICE)
                    loss = self.loss(scores, batch_label)
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    total_loss += loss.item()
                    vali_loss = self.eval_on_vali()
                progress.set_description(f"epoch{ep} train loss: {total_loss}, vali loss: {vali_loss}")
                torch.save(self.model.state_dict(), self.args.save_dir)
    
    def train_epoch(self):
        self.model.train()
        pass

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
    # print(feature['test'].text)

    model = FruadDection
    # model.forward(list(feature['test'].text)[:100])
    trainer = FraudDectionTrainer(model, feature, label, args)
    # if DEBUG:
    #     author_ids, title_embeddings, text_sentences_embeddings = trainer.convert_text_to_tensor(trainer.vali_feature[1:10], tag='train', cache=False)
    # else:
    #     author_ids, title_embeddings, text_sentences_embeddings = trainer.convert_text_to_tensor(trainer.train_feature, tag='train', cache=False)
    author_emb_dim = 32
    sentence_embed_dim = 768
    trainer.initialize(author_emb_dim, sentence_embed_dim)
    trainer.train()

    # se = SentenceEmbedding()
    # se.get_sentence_embeddings(None)
