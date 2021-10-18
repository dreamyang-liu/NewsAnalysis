from tqdm import tqdm
from FDModel import *
from Data import *

import torch
import multiprocessing
import pysbd

def collect_all_authors(authors_list):
    res = set()
    author_res = []
    for authors in authors_list:
        author_res.append(authors)
        for author in authors:
            res.add(author)
            if len(author) > 50:
                print(author)
    return list(res), author_res

def collect_all_titles(titles_list):
    segmenter = pysbd.Segmenter()
    res = []
    for titles in tqdm(titles_list):
        res.append(segmenter.segment(titles))
    return res

def collect_all_text_worker(text_list, return_list, i):
    segmenter = pysbd.Segmenter()
    res = []
    for texts in tqdm(text_list):
        res.append(segmenter.segment(texts))
        if segmenter.segment(texts).__len__() == 0:
            print(len(texts), texts == ' ', texts == '  ', texts == '   ')
    return_list[i] = res

def collect_all_text(texts_list, process=1):
    manager = multiprocessing.Manager()
    return_list = manager.dict()
    batch_size = int(len(texts_list) / process) + 1
    jobs = []
    get_batch = lambda x: texts_list[batch_size * x:batch_size * x + batch_size]
    for i in range(process):
        p = multiprocessing.Process(target=collect_all_text_worker, args=(get_batch(i), return_list, i))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    res = []
    for i in range(process):
        res.extend(return_list[i])
    return res

class DataGenerator(object):

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        _feature = self.feature['train']
        author_set, _ = collect_all_authors(_feature['author'])
        self.author_tokenizer = AuthorTokenizer(author_set)
        self.author_embedding = AuthorEmbedding(self.author_tokenizer.author_length, 32).to(TRAIN_DEVICE)
        self.sentence_embedding = SentenceEmbedding(device=PRETRAIN_DEVICE, output_device=TRAIN_DEVICE)
    
    def get_train_features(self):
        _feature = self.feature['train']
        _label = torch.tensor(self.label['train'].values, dtype=torch.long)
        _, author_list = collect_all_authors(_feature['author'])
        title_list = collect_all_titles(_feature['title'])
        text_list = collect_all_text(_feature['text'], 30)
        author_emb = [self.author_embedding(self.author_tokenizer.encode_author(author).to(TRAIN_DEVICE)) for author in author_list]
        title_emb = [self.sentence_embedding.get_sentence_embeddings(title) for title in title_list]
        text_emb = [self.sentence_embedding.get_sentence_embeddings(text) for text in text_list]
        return author_emb, title_emb, text_emb, _label
    
    def get_test_features(self):
        _feature = self.feature['test']
        _label = torch.tensor(self.label['test'].values, dtype=torch.long)
        _, author_list = collect_all_authors(_feature['author'])
        title_list = collect_all_titles(_feature['title'])
        text_list = collect_all_text(_feature['text'], 30)
        author_emb = (self.author_embedding(torch.tensor(self.author_tokenizer.encode_author(author), dtype=torch.long)) for author in author_list)
        title_emb = (self.sentence_embedding.get_sentence_embeddings(title) for title in title_list)
        text_emb = (self.sentence_embedding.get_sentence_embeddings(text) for text in text_list)
        return author_emb, title_emb, text_emb, _label




if __name__ == "__main__":
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)

    feature, label = dp.default_process(split_dataset=True)
