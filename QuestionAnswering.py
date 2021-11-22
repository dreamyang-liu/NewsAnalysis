from transformers import DistilBertForQuestionAnswering
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, AutoModel
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from datasets import load_dataset
import numpy
from transformers import AutoModelForQuestionAnswering
from transformers import DistilBertTokenizerFast
from transformers import pipeline
import torchvision
import wget
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

"""
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class QuestionAnsweringTrainer(object):

    def __init__(self):
        self.choice = "distilbert"
        self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def read_squad(path):
        path = Path(path)
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
        return contexts, questions, answers

    def add_end_idx(answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

    def add_token_positions(encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start'], sequence_index = 1))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1, sequence_index = 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    def preprocess(self):
        wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json', 'train-v2.0.json')
        wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json', 'dev-v2.0.json')
        train_contexts, train_questions, train_answers = read_squad('train-v2.0.json')
        val_contexts, val_questions, val_answers = read_squad('dev-v2.0.json')
        add_end_idx(train_answers, train_contexts)
        add_end_idx(val_answers, val_contexts)
        tokenizer = self.tokenizer
        train_encodings = tokenizer(text = train_questions, text_pair = train_contexts, truncation="only_second", padding=True)
        val_encodings = tokenizer(text = val_questions, text_pair = val_contexts, truncation="only_second", padding=True)
        add_token_positions(train_encodings, train_answers)
        add_token_positions(val_encodings, val_answers)
        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)


    def train(self):
        model = self.model
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.train()
        optim = AdamW(model.parameters(), lr=5e-5)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        for epoch in range(3):
            model.train()
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())



        self.model = model


    def evaluate(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=16)
        acc = []
        for batch in val_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_true = batch['start_positions'].to(device)
                end_true = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                start_pred = torch.argmax(outputs['start_logits'], dim=1)
                end_pred = torch.argmax(outputs['end_logits'], dim=1)
                acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
                acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
        acc = sum(acc)/len(acc)
        print(acc)

"""

class QuestionAnsweringModel(object):
    def __init__(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained('aszidon/' + 'distilbert' + 'custom5')
        self.tokenizer = AutoTokenizer.from_pretrained('aszidon/' + 'distilbert' + 'custom5')
        self.pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="pt")

    def getanswer(quest, cont):
        return self.pipe(question=quest, context=cont)['answer']
