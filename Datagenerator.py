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

def collect_all_text_worker(text_list, return_list, idx):
    segmenter = pysbd.Segmenter()
    res = []
    for texts in tqdm(text_list):
        sentences = segmenter.segment(texts)
        if len(sentences) >= 400:
            merge_sentences = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    merge_sentences.append(sentences[i] + sentences[i+1])
                else:
                    merge_sentences.append(sentences[i])
            res.append(merge_sentences)
        else:
            res.append(sentences[:1000])
    return_list[idx] = res

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

    def __init__(self, feature, label, process=1):
        self.feature = feature
        self.label = label
        _feature = self.feature['train']
        author_set, _ = collect_all_authors(_feature['author'])
        self.process = process
        self.author_tokenizer = AuthorTokenizer(author_set)
        self.author_embedding = AuthorEmbedding(self.author_tokenizer.author_length, 32).to(TRAIN_DEVICE)
        self.sentence_embedding = SentenceEmbedding(device=PRETRAIN_DEVICE, output_device=TRAIN_DEVICE)
    
    def get_eval_features(self, feature):
        _feature = feature
        text_list = collect_all_text(_feature['text'], self.process)
        _, author_list = collect_all_authors(_feature['author'])
        title_list = collect_all_titles(_feature['title'])
        author_emb = [self.author_embedding(self.author_tokenizer.encode_author(author).to(TRAIN_DEVICE)) for author in author_list]
        title_emb = [self.sentence_embedding.get_sentence_embeddings(title) for title in title_list]
        text_emb = [self.sentence_embedding.get_sentence_embeddings(text) for text in tqdm(text_list)]
        return author_emb, title_emb, text_emb
    
    def get_train_features(self):
        _feature = self.feature['train']
        _label = torch.tensor(self.label['train'].values, dtype=torch.long)
        text_list = collect_all_text(_feature['text'], self.process)
        _, author_list = collect_all_authors(_feature['author'])
        title_list = collect_all_titles(_feature['title'])
        author_emb = [self.author_embedding(self.author_tokenizer.encode_author(author).to(TRAIN_DEVICE)) for author in author_list]
        title_emb = [self.sentence_embedding.get_sentence_embeddings(title) for title in title_list]
        text_emb = [self.sentence_embedding.get_sentence_embeddings(text) for text in tqdm(text_list)]
        return author_emb, title_emb, text_emb, _label
    
    def get_test_features(self):
        _feature = self.feature['vali']
        _label = torch.tensor(self.label['vali'].values, dtype=torch.long)
        text_list = collect_all_text(_feature['text'], self.process)
        _, author_list = collect_all_authors(_feature['author'])
        title_list = collect_all_titles(_feature['title'])
        author_emb = [self.author_embedding(self.author_tokenizer.encode_author(author).to(TRAIN_DEVICE)) for author in author_list]
        title_emb = [self.sentence_embedding.get_sentence_embeddings(title) for title in title_list]
        text_emb = [self.sentence_embedding.get_sentence_embeddings(text) for text in tqdm(text_list)]
        return author_emb, title_emb, text_emb, _label
    
    def get_tensor(self, author, title, text, device=torch.device('cpu')):
        author_emb = self.author_embedding(self.author_tokenizer.encode_author(author).to(device))
        title_emb = self.sentence_embedding.get_sentence_embeddings(title).to(device)
        text_emb = self.sentence_embedding.get_sentence_embeddings(text).to(device)
        return author_emb, title_emb, text_emb




if __name__ == "__main__":
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)

    feature, label = dp.default_process(split_dataset=True)
