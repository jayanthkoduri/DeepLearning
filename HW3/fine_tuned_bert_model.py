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

device_setting = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def read_dataset(json_file_path):

    doc_texts = []
    query_texts = []
    solution_texts = []

    total_queries = 0
    total_with_answers = 0
    total_without_answers = 0

    with open(json_file_path, 'r') as file_handle:
        dataset_json = json.load(file_handle)

    for item in dataset_json['data']:
        for passage in item['paragraphs']:
            passage_text = passage['context'].lower()
            for q_and_a in passage['qas']:
                total_queries += 1
                query_text = q_and_a['question'].lower()
                for solution in q_and_a['answers']:
                    doc_texts.append(passage_text)
                    query_texts.append(query_text)
                    solution_texts.append(solution)

    return total_queries, total_with_answers, total_without_answers, doc_texts, query_texts, solution_texts

total_queries_train, with_answers_train, without_answers_train, docs_train, queries_train, solutions_train = read_dataset('data/spoken_train-v1.1.json')
query_count_train = total_queries_train

total_queries_valid, with_answers_valid, without_answers_valid, docs_valid, queries_valid, solutions_valid = read_dataset('data/spoken_test-v1.1.json')
query_count_valid = total_queries_valid

def add_solution_end(solutions, docs):
    for solution, _ in zip(solutions, docs):
        solution['text'] = solution['text'].lower()
        solution['solution_end'] = solution['answer_start'] + len(solution['text'])
add_solution_end(solutions_train, docs_train)
add_solution_end(solutions_valid, docs_valid)

MAX_SEQ_LENGTH = 512
PRETRAINED_MODEL = "deepset/bert-base-cased-squad2"

stride_len = 128
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
padding_side = tokenizer.padding_side == "right"

def truncate_docs_tokenize(docs, solutions, queries, tokenizer, max_length, stride_length):
    truncated_docs = []
    for i, doc in enumerate(docs):
        if len(doc) > max_length:
            solution_start = solutions[i]['answer_start']
            solution_end = solution_start + len(solutions[i]['text'])
            center_point = (solution_start + solution_end) // 2
            initial_point = max(0, min(center_point - max_length // 2, len(doc) - max_length))
            final_point = initial_point + max_length
            initial_point = int(initial_point)
            final_point = int(final_point)
            truncated_doc = doc[initial_point:final_point]
            truncated_docs.append(truncated_doc)
            solutions[i]['answer_start'] = solution_start - initial_point
        else:
            truncated_docs.append(doc)

    doc_encodings = tokenizer(
        queries,
        truncated_docs,
        max_length=max_length,
        truncation=True,
        stride=stride_length,
        padding='max_length',
        return_tensors='pt'
    )
    return doc_encodings

training_encodings = truncate_docs_tokenize(docs_train, solutions_train, queries_train, tokenizer, MAX_SEQ_LENGTH, stride_len)
validation_encodings = tokenizer(queries_valid, docs_valid, max_length=MAX_SEQ_LENGTH, truncation=True, stride=stride_len, padding='max_length', return_tensors='pt')

def locate_solution_positions_train(index):
    start_position = 0
    end_position = 0
    solution_encoding = tokenizer(solutions_train[index]['text'],  max_length = MAX_SEQ_LENGTH, truncation=True, padding=True)
    for offset in range( len(training_encodings['input_ids'][index]) -  len(solution_encoding['input_ids']) ):
        sequence_match = True
        for inner_idx in range(1,len(solution_encoding['input_ids']) - 1):
            if (solution_encoding['input_ids'][inner_idx] != training_encodings['input_ids'][index][offset + inner_idx]):
                sequence_match = False
                break
            if sequence_match:
                start_position = offset+1
                end_position = offset+inner_idx+1
                break
    return(start_position, end_position)

start_locs = []
end_locs = []
mismatch_counter = 0
for idx in range(len(training_encodings['input_ids'])):
    loc_start, loc_end = locate_solution_positions_train(idx)
    start_locs.append(loc_start)
    end_locs.append(loc_end)
    if loc_start==0:
        mismatch_counter += 1
training_encodings.update({'start_positions': start_locs, 'end_positions': end_locs})

def locate_solution_positions_valid(index):
    start_position = 0
    end_position = 0
    solution_encoding = tokenizer(solutions_valid [index]['text'],  max_length = MAX_SEQ_LENGTH, truncation=True, padding=True)
    for offset in range( len(validation_encodings['input_ids'][index]) -  len(solution_encoding['input_ids']) ):
        sequence_match = True
        for inner_idx in range(1,len(solution_encoding['input_ids']) - 1):
            if (solution_encoding['input_ids'][inner_idx] != validation_encodings['input_ids'][index][offset + inner_idx]):
                sequence_match = False
                break
            if sequence_match:
                start_position = offset+1
                end_position = offset+inner_idx+1
                break
    return(start_position, end_position)

start_locs = []
end_locs = []
mismatch_counter = 0
for idx in range(len(validation_encodings['input_ids'])):
    loc_start, loc_end = locate_solution_positions_valid(idx)
    start_locs.append(loc_start)
    end_locs.append(loc_end)
    if loc_start==0:
        mismatch_counter += 1

validation_encodings.update({'start_positions': start_locs, 'end_positions': end_locs})

class EncodedDataset(Dataset):
    def __init__(self, encoded_pairs):
        self.encoded_pairs = {key: torch.tensor(value) for key, value in encoded_pairs.items()}
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encoded_pairs.items()}
    
    def __len__(self):
        return len(self.encoded_pairs['input_ids'])
training_dataset = EncodedDataset(training_encodings)
validation_dataset = EncodedDataset(validation_encodings)

training_loader = DataLoader(training_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16)

transformer_model = BertModel.from_pretrained(PRETRAINED_MODEL)

import torch
import torch.nn.functional as F 
class TransformerQAModel(nn.Module):
    def __init__(self, transformer_model_name='deepset/bert-base-cased-squad2'):
        super().__init__()
        self.transformer = BertModel.from_pretrained(transformer_model_name)
        self.drop_layer = nn.Dropout(0.1)
        self.linear_layer1 = nn.Linear(768 * 2, 768 * 2)
        self.output_layer = nn.Linear(768 * 2, 2)
        self.activation_func = nn.LeakyReLU()

    def forward(self, input_id_stream, attention_mask_stream, token_type_id_stream):
        model_outputs = self.transformer(input_id_stream, attention_mask=attention_mask_stream, token_type_ids=token_type_id_stream, return_dict=True, output_hidden_states=True)
        concat_hidden_states = torch.cat((model_outputs.hidden_states[-1], model_outputs.hidden_states[-3]), dim=-1)
        x = self.drop_layer(concat_hidden_states)
        x = self.activation_func(self.linear_layer1(x))
        logits = self.output_layer(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

transformer_qa_model = TransformerQAModel(transformer_model_name='deepset/bert-base-cased-squad2')

def calculate_focal_loss(start_logit_stream, end_logit_stream, start_pos_stream, end_pos_stream, focal_gamma=2.0):
    ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    start_loss_vals = ce_loss_func(start_logit_stream, start_pos_stream)
    end_loss_vals = ce_loss_func(end_logit_stream, end_pos_stream)
    
    start_probs = F.softmax(start_logit_stream, dim=1).gather(1, start_pos_stream.unsqueeze(1)).squeeze(1)
    end_probs = F.softmax(end_logit_stream, dim=1).gather(1, end_pos_stream.unsqueeze(1)).squeeze(1)
    
    focal_loss_start = ((1 - start_probs) ** focal_gamma) * start_loss_vals
    focal_loss_end = ((1 - end_probs) ** focal_gamma) * end_loss_vals
    
    focal_loss_total = (focal_loss_start + focal_loss_end) / 2.0
    
    return focal_loss_total.mean()

optimizer = AdamW(transformer_qa_model.parameters(), lr=2e-5, weight_decay=2e-2)
scheduler = ExponentialLR(optimizer, gamma=0.9)

def train_model_epoch(model, loader, current_epoch):
    model.train()
    epoch_losses = []
    epoch_accuracy = []
    progress_indicator = 0

    for batch_data in tqdm(loader, desc='Epoch ' + str(current_epoch) + ' Progress:'):
        optimizer.zero_grad()
        input_ids_batch = batch_data['input_ids'].to(device_setting)
        attention_masks_batch = batch_data['attention_mask'].to(device_setting)
        token_type_ids_batch = batch_data['token_type_ids'].to(device_setting)
        start_pos_batch = batch_data['start_positions'].to(device_setting)
        end_pos_batch = batch_data['end_positions'].to(device_setting)
        
        start_predictions, end_predictions = model(input_id_stream=input_ids_batch, attention_mask_stream=attention_masks_batch, token_type_id_stream=token_type_ids_batch)

        loss = calculate_focal_loss(start_predictions, end_predictions, start_pos_batch, end_pos_batch, 1)  # Using gamma=1 for focal loss
        epoch_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        start_pred_class = torch.argmax(start_predictions, dim=1)
        end_pred_class = torch.argmax(end_predictions, dim=1)
        accuracy = ((start_pred_class == start_pos_batch).float().mean() + (end_pred_class == end_pos_batch).float().mean()) / 2.0
        epoch_accuracy.append(accuracy.item())

        if progress_indicator == 250 and current_epoch == 1:
            print(f'Intermediate Loss after 250 batches in epoch 1: {sum(epoch_losses)/len(epoch_losses)}')
            print(f'Intermediate Accuracy after 250 batches in epoch 1: {sum(epoch_accuracy)/len(epoch_accuracy)}')
            progress_indicator = 0
        
        progress_indicator += 1

    scheduler.step()
    epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
    epoch_avg_acc = sum(epoch_accuracy) / len(epoch_accuracy)

    return epoch_avg_acc, epoch_avg_loss

def evaluate_model(model, loader):
    model.eval()
    prediction_comparison = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluation in Progress'):
            input_ids_eval = batch['input_ids'].to(device_setting)
            attention_mask_eval = batch['attention_mask'].to(device_setting)
            token_type_ids_eval = batch['token_type_ids'].to(device_setting)

            start_logits_eval, end_logits_eval = model(input_id_stream=input_ids_eval, attention_mask_stream=attention_mask_eval, token_type_id_stream=token_type_ids_eval)

            for idx in range(input_ids_eval.shape[0]):
                start_pred_idx = torch.argmax(start_logits_eval[idx]).item()
                end_pred_idx = torch.argmax(end_logits_eval[idx]).item() + 1  # +1 to include the end token
                predicted_answer = tokenizer.decode(input_ids_eval[idx][start_pred_idx:end_pred_idx], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                actual_answer = tokenizer.decode(input_ids_eval[idx][batch['start_positions'][idx]:batch['end_positions'][idx]+1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                prediction_comparison.append((predicted_answer, actual_answer))

    return prediction_comparison

import evaluate
from evaluate import load

wer_metric = load("wer")
EPOCH_COUNT = 4
transformer_qa_model.to(device_setting)
wer_scores_over_epochs = []

for epoch_num in range(EPOCH_COUNT):
    training_accuracy, training_loss = train_model_epoch(transformer_qa_model, training_loader, epoch_num+1)
    print(f'Epoch - {epoch_num+1}')
    print(f'Accuracy: {training_accuracy}')
    print(f'Loss: {training_loss}')

    eval_predictions = evaluate_model(transformer_qa_model, validation_loader)

    predicted_texts = [pair[0] if len(pair[0]) > 0 else "$" for pair in eval_predictions]
    actual_texts = [pair[1] if len(pair[1]) > 0 else "$" for pair in eval_predictions]

    wer_score_epoch = wer_metric.compute(predictions=predicted_texts, references=actual_texts)
    wer_scores_over_epochs.append(wer_score_epoch)

print('WER (fine-tuned model) - ', wer_scores_over_epochs)
