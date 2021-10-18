import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.map = bidict({author:i+2 for i, author in enumerate(set(author_list))})
        self.map['Anonymous'] = 0
        self.map['Author Sentence'] = 1
        self.author_length = set(author_list).__len__() + 2
    
    def assign(self, author):
        if author in dict(self.map).keys():
            return self.map[author]
        elif len(author) >= 50:
            return self.map['Author Sentence']
        else:
            return 0
    
    def encode_author(self, authors):
        if isinstance(authors, str):
            return torch.tensor([self.assign(authors)], dtype=torch.long)
        if isinstance(authors, list):
            return torch.tensor([self.assign(x) for x in authors], dtype=torch.long)


class SentenceEmbedding(object):

    def __init__(self, device=PRETRAIN_DEVICE, output_device=PRETRAIN_DEVICE):
        self.device = device
        self.output_device = output_device
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2').to(self.device)
        freeze_model(self.model)
    
    def get_sentence_embeddings(self, sentences):
        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        # Compute token embeddings
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.to(self.output_device)
        gpu_memory_collect()
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
        self.context = Variable(torch.randn((1, attn_emb_dim), requires_grad = True))
    
    def forward(self, X):
        """
        X: K * emb_dim
        """
        o = self.fc(X) # K * attn_emb_dim
        o = F.tanh(o)
        o = self.context @ o.t() # 1 * K
        o = F.softmax(o)
        return o @ X

class FDModel(nn.Module):

    def __init__(self, author_dim, sentence_dim):
        super(FDModel, self).__init__()
        self._author_attn = Attention(author_dim, 32)
        self._title_attn = Attention(sentence_dim, 32)
        self._text_attn = Attention(sentence_dim, 32)

        self._author_fc = nn.Linear(author_dim, 1)
        self._title_fc = nn.Linear(sentence_dim, 1)
        self._text_fc = nn.Linear(sentence_dim, 1)

        self._classifier_fc = nn.Linear(3, 1)
    
    def forward(self, author_emb, title_emb, text_emb):
        o_attn_author = self._author_attn(author_emb)
        o_attn_title = self._title_attn(title_emb)
        o_attn_text = self._text_attn(text_emb)

        score_fc_author = self._author_fc(o_attn_author)
        score_fc_title = self._title_fc(o_attn_title)
        score_fc_text = self._text_fc(o_attn_text)

        score = F.sigmoid(torch.cat([score_fc_author, score_fc_title, score_fc_text]))
        return F.sigmoid(self._classifier_fc(score))


if __name__ == '__main__':
    pass