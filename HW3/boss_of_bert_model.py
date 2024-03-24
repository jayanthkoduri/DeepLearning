#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import transformers
from transformers import BertModel, BertTokenizerFast
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print(device)


# In[3]:


def load_dataset(file_path):
    """
    Load and preprocess dataset from a JSON file.

    Parameters:
    - file_path: Path to the JSON file containing the dataset.

    Returns:
    - num_questions: Total number of questions.
    - num_positive: Total number of positive answers (Not used in this function, but placeholder for extension).
    - num_impossible: Total number of impossible questions (Not used in this function, but placeholder for extension).
    - contexts: List of context passages.
    - questions: List of questions.
    - answers: List of answer dictionaries.
    """
    contexts = []
    questions = []
    answers = []

    num_questions = 0
    num_positive = 0  # positive answers count
    num_impossible = 0  # impossible questions count

    with open(file_path, 'r') as file:
        data = json.load(file)

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context'].lower()
            for qa in paragraph['qas']:
                num_questions += 1
                question = qa['question'].lower()
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return num_questions, num_positive, num_impossible, contexts, questions, answers


# In[4]:


# Load and preprocess the training dataset
num_q, num_pos, num_imp, train_contexts, train_questions, train_answers = load_dataset('data/spoken_train-v1.1.json')
num_questions_train = num_q
num_possible_train = num_pos
num_impossible_train = num_imp

# Load and preprocess the validation dataset
num_q, num_pos, num_imp, valid_contexts, valid_questions, valid_answers = load_dataset('data/spoken_test-v1.1.json')
num_questions_valid = num_q
num_possible_valid = num_pos
num_impossible_valid = num_imp


# In[5]:


def append_answer_end_to_answers(answers, contexts):
    """
    Enhance the answers dictionary by appending the 'answer_end' key.

    Parameters:
    - answers: A list of dictionaries, where each dictionary represents an answer and contains at least 'text' and 'answer_start' keys.
    - contexts: A list of context strings corresponding to each answer in the answers list.

    Returns:
    None; the function modifies the answers list in place.
    """
    for answer, _ in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

#training and validation datasets
append_answer_end_to_answers(train_answers, train_contexts)
append_answer_end_to_answers(valid_answers, valid_contexts)


# In[6]:


MAX_LENGTH = 512
MODEL_PATH = "bert-base-uncased"

doc_stride = 128
tokenizerFast = BertTokenizerFast.from_pretrained(MODEL_PATH)
pad_on_right = tokenizerFast.padding_side == "right"
train_contexts_trunc=[]


# In[7]:


# train_encodings = tokenizerFast(train_questions, train_contexts,  max_length = MAX_LENGTH,truncation=True,padding=True)
# valid_encodings = tokenizerFast(valid_questions,valid_contexts,  max_length = MAX_LENGTH, truncation=True,padding=True)


# In[8]:


def truncate_and_tokenize_contexts(contexts, answers, questions, tokenizer, max_length, doc_stride):
    truncated_contexts = []
    for i, context in enumerate(contexts):
        if len(context) > max_length:
            answer_start = answers[i]['answer_start']
            answer_end = answer_start + len(answers[i]['text'])
            mid_point = (answer_start + answer_end) // 2
            start_point = max(0, min(mid_point - max_length // 2, len(context) - max_length))
            end_point = start_point + max_length
            start_point = int(start_point)
            end_point = int(end_point)
            truncated_context = context[start_point:end_point]
            truncated_contexts.append(truncated_context)
            answers[i]['answer_start'] = answer_start - start_point
        else:
            truncated_contexts.append(context)

    encodings = tokenizer(
        questions,
        truncated_contexts,
        max_length=max_length,
        truncation=True,
        stride=doc_stride,
        padding='max_length',
        return_tensors='pt'
    )
    return encodings

train_encodings_fast = truncate_and_tokenize_contexts(train_contexts, train_answers, train_questions, tokenizerFast, MAX_LENGTH, doc_stride)
valid_encodings_fast = tokenizerFast(valid_questions, valid_contexts, max_length=MAX_LENGTH, truncation=True, stride=doc_stride, padding='max_length', return_tensors='pt')


# In[9]:


def ret_Answer_start_and_end_train(idx):
    ret_start = 0
    ret_end = 0
    answer_encoding_fast = tokenizerFast(train_answers[idx]['text'],  max_length = MAX_LENGTH, truncation=True, padding=True)
    for a in range( len(train_encodings_fast['input_ids'][idx]) -  len(answer_encoding_fast['input_ids']) ):
        match = True
        for i in range(1,len(answer_encoding_fast['input_ids']) - 1):
            if (answer_encoding_fast['input_ids'][i] != train_encodings_fast['input_ids'][idx][a + i]):
                match = False
                break
            if match:
                ret_start = a+1
                ret_end = a+i+1
                break
    return(ret_start, ret_end)

start_positions = []
end_positions = []
ctr = 0
for h in range(len(train_encodings_fast['input_ids'])):
    s, e = ret_Answer_start_and_end_train(h)
    start_positions.append(s)
    end_positions.append(e)
    if s==0:
        ctr = ctr + 1
    
train_encodings_fast.update({'start_positions': start_positions, 'end_positions': end_positions})
valid_encodings_fast.update({'start_positions': start_positions, 'end_positions': end_positions})


# In[11]:


class InputDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])
train_dataset = InputDataset(train_encodings_fast)
valid_dataset = InputDataset(valid_encodings_fast)

train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=16)


# In[12]:


bert_model = BertModel.from_pretrained(MODEL_PATH)


# In[13]:


import torch
import torch.nn as nn
from transformers import BertModel

class QAModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 * 2, 768 * 2)
        self.fc2 = nn.Linear(768 * 2, 2)
        self.relu = nn.LeakyReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
        concatenated_hidden_states = torch.cat((outputs.hidden_states[-1], outputs.hidden_states[-3]), dim=-1)
        x = self.dropout(concatenated_hidden_states)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

model = QAModel(bert_model_name='bert-base-uncased')


# In[14]:


import torch.nn.functional as F
def compute_focal_loss(start_logits, end_logits, start_positions, end_positions, gamma=2.0):
    """
    Computes the focal loss for both start and end positions of answers in a question-answering model.

    Parameters:
    - start_logits (torch.Tensor): Logits for start positions, shape [batch_size, seq_length].
    - end_logits (torch.Tensor): Logits for end positions, shape [batch_size, seq_length].
    - start_positions (torch.Tensor): True start positions, shape [batch_size].
    - end_positions (torch.Tensor): True end positions, shape [batch_size].
    - gamma (float): Focusing parameter for focal loss to reduce the loss contribution from easy examples and put more focus on hard, misclassified examples.

    Returns:
    - torch.Tensor: The average focal loss for start and end logits.
    """
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    
    start_probs = F.softmax(start_logits, dim=1).gather(1, start_positions.unsqueeze(1)).squeeze(1)
    end_probs = F.softmax(end_logits, dim=1).gather(1, end_positions.unsqueeze(1)).squeeze(1)
    
    start_focal_loss = ((1 - start_probs) ** gamma) * start_loss
    end_focal_loss = ((1 - end_probs) ** gamma) * end_loss
    
    focal_loss = (start_focal_loss + end_focal_loss) / 2.0
    
    return focal_loss.mean()


# In[15]:


optim = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)
total_acc = []
total_loss = []


# In[16]:


from tqdm import tqdm
import torch

def train_epoch(model, dataloader, epoch):
    model.train()
    total_losses = []
    total_acc = []
    batch_tracker = 0

    for batch in tqdm(dataloader, desc='Running Epoch ' + str(epoch) + ':'):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        out_start, out_end = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = compute_focal_loss(out_start, out_end, start_positions, end_positions, 1)  # Gamma=1
        total_losses.append(loss.item())
        
        loss.backward()
        optim.step()
        
        start_pred = torch.argmax(out_start, dim=1)
        end_pred = torch.argmax(out_end, dim=1)
        acc = ((start_pred == start_positions).float().mean() + (end_pred == end_positions).float().mean()) / 2.0
        total_acc.append(acc.item())

        if batch_tracker == 250 and epoch == 1:
            print(f'Intermediate Loss after 250 batches in epoch 1: {sum(total_losses)/len(total_losses)}')
            print(f'Intermediate Accuracy after 250 batches in epoch 1: {sum(total_acc)/len(total_acc)}')
            batch_tracker = 0
        
        batch_tracker += 1

    avg_loss = sum(total_losses) / len(total_losses)
    avg_acc = sum(total_acc) / len(total_acc)

    return avg_acc, avg_loss


# In[17]:


def eval_model(model, dataloader):
    model.eval()
    answer_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            out_start, out_end = model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

            for i in range(input_ids.shape[0]):
                start_pred = torch.argmax(out_start[i]).item()
                end_pred = torch.argmax(out_end[i]).item() + 1 
                pred_answer = tokenizerFast.decode(input_ids[i][start_pred:end_pred], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                true_answer = tokenizerFast.decode(input_ids[i][start_positions[i]:end_positions[i]+1], skip_special_tokens=True, clean_up_tokenization_spaces=True)  # +1 to make end position inclusive
                
                answer_list.append((pred_answer, true_answer))

    return answer_list


# In[19]:


import evaluate
from evaluate import load

wer = load("wer")
EPOCHS = 4
model.to(device)
wer_list = []

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_data_loader, epoch+1)
    print(f'Epoch - {epoch+1}')
    print(f'Accuracy: {train_acc}')
    print(f'Loss: {train_loss}')

    answer_list = eval_model(model, valid_data_loader)

    pred_answers = [ans[0] if len(ans[0]) > 0 else "$" for ans in answer_list]
    true_answers = [ans[1] if len(ans[1]) > 0 else "$" for ans in answer_list]

    wer_score = wer.compute(predictions=pred_answers, references=true_answers)
    wer_list.append(wer_score)

print('Boss: ', wer_list)