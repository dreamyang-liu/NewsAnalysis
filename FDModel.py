import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch.autograd import Variable
from transformers import AutoTokenizer, AutoModel
from argparser import *
from utils import *
from bidict import bidict

class AuthorEmbedding(nn.Module):

    def __init__(self, author_dim, embed_dim):
        super(AuthorEmbedding, self).__init__()
        self._embedding = nn.Embedding(author_dim, embed_dim)
    
    def forward(self, inputs):
        return self._embedding(inputs)

class AuthorTokenizer(object):
    
    def __init__(self, author_list):
        self.map = bidict({author:i+3 for i, author in enumerate(set(author_list))})
        self.map['Anonymous'] = 0
        self.map['Author Sentence'] = 1
        self.map['Unknown'] = 2
        self.author_length = set(author_list).__len__() + 3
    
    def assign(self, author):
        if author in dict(self.map).keys():
            return self.map[author]
        elif len(author) >= 50:
            return self.map['Author Sentence']
        else:
            return self.map['Unknown']
    
    def encode_author(self, authors):
        if isinstance(authors, str):
            return torch.tensor([self.assign(authors)], dtype=torch.long)
        if isinstance(authors, list):
            return torch.tensor([self.assign(x) for x in authors], dtype=torch.long)


class SentenceEmbedding(object):

    def __init__(self, device=PRETRAIN_DEVICE, output_device=TRAIN_DEVICE):
        self.device = device
        self.output_device = output_device
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2').to(self.device)
        freeze_model(self.model)
    
    def get_sentence_embeddings(self, sentences):
        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']
        # raise ValueError(f'Debug break: {len(sentences)}')
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        # Compute token embeddings
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.to(self.output_device)
        return sentence_embeddings
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Attention(nn.Module):

    def __init__(self, emb_dim, attn_emb_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(emb_dim, attn_emb_dim)
        self.context = nn.Parameter(torch.randn((1, attn_emb_dim))).to(TRAIN_DEVICE)
    
    def forward(self, X):
        """
        X: K * emb_dim
        """
        o = self.fc(X) # K * attn_emb_dim
        o = torch.tanh(o)
        o = self.context @ o.t() # 1 * K
        o = F.softmax(o, dim=0)
        return o @ X

class FDModel(nn.Module):

    def __init__(self, author_dim, sentence_dim):
        super(FDModel, self).__init__()
        self.author_dim = author_dim
        self.sentence_dim = sentence_dim
        self._author_attn = Attention(author_dim, 32)
        self._title_attn = Attention(sentence_dim, 32)
        self._text_attn = Attention(sentence_dim, 32)

        self._author_fc = nn.Linear(author_dim, 1)
        self._title_fc = nn.Linear(sentence_dim, 1)
        self._text_fc = nn.Linear(sentence_dim, 1)

        self._classifier_fc = nn.Linear(3, 2)
    
    def forward(self, author_emb, title_emb, text_emb):
        batch_size = len(author_emb)
        batch_attn_author = []
        batch_attn_title = []
        batch_attn_text = []

        for i in range(batch_size):
            batch_attn_author.append(self._author_attn(author_emb[i]))
            batch_attn_title.append(self._title_attn(title_emb[i]))
            batch_attn_text.append(self._text_attn(text_emb[i]))
        
        # pdb.set_trace()
        batch_attn_author = torch.cat(batch_attn_author, dim=0)
        batch_attn_title = torch.cat(batch_attn_title, dim=0)
        batch_attn_text = torch.cat(batch_attn_text, dim=0)

        score_fc_author = self._author_fc(batch_attn_author)
        score_fc_title = self._title_fc(batch_attn_title)
        score_fc_text = self._text_fc(batch_attn_text)

        score = torch.sigmoid(torch.cat([score_fc_author, score_fc_title, score_fc_text], dim=1))
        return F.softmax(self._classifier_fc(score), dim=1)
    
    def forward_single(self, author_emb, title_emb, text_emb):

        attn_author = self._author_attn(author_emb)
        attn_title = self._title_attn(title_emb)
        attn_text = self._text_attn(text_emb)

        score_fc_author = self._author_fc(attn_author)
        score_fc_title = self._title_fc(attn_title)
        score_fc_text = self._text_fc(attn_text)

        score = torch.sigmoid(torch.cat([score_fc_author, score_fc_title, score_fc_text], dim=1))
        return F.softmax(self._classifier_fc(score), dim=1)


if __name__ == '__main__':
    pass