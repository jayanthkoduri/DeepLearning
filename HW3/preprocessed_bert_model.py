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

# Setup device
processing_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Function to load and process dataset
def read_and_process_dataset(dataset_file_path):
    """
    Reads and processes a dataset from a JSON file.

    Parameters:
    - dataset_file_path: Path to the JSON file with the dataset.

    Returns:
    - total_questions: Number of questions in the dataset.
    - total_positive_answers: Number of questions with positive answers (for potential future use).
    - total_impossible_questions: Number of impossible questions (for potential future use).
    - passage_texts: List of context passages.
    - query_texts: List of questions.
    - answer_info: List of answer dictionaries.
    """
    passage_texts = []
    query_texts = []
    answer_info = []

    total_questions = 0
    total_positive_answers = 0
    total_impossible_questions = 0

    with open(dataset_file_path, 'r') as file:
        dataset_json = json.load(file)

    for article in dataset_json['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context'].lower()
            for qa in paragraph['qas']:
                total_questions += 1
                question = qa['question'].lower()
                for answer in qa['answers']:
                    passage_texts.append(context)
                    query_texts.append(question)
                    answer_info.append(answer)

    return total_questions, total_positive_answers, total_impossible_questions, passage_texts, query_texts, answer_info

# Add 'answer_end' information to answer dictionaries
def enhance_answers_with_end(answer_data, passages):
    """
    Modifies the answer dictionaries in place, adding 'answer_end' key.

    Parameters:
    - answer_data: List of dictionaries, each representing an answer.
    - passages: List of context strings associated with each answer.

    Returns:
    None; modifies the answer_data list in place.
    """
    for answer, passage in zip(answer_data, passages):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

# Constants for preprocessing
MAX_LENGTH = 512
MODEL_PATH = "bert-base-uncased"
doc_stride = 128
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

# Preprocess datasets
training_dataset_path = 'data/spoken_train-v1.1.json'
validation_dataset_path = 'data/spoken_test-v1.1.json'

# Load and process datasets
total_questions_train, _, _, train_passages, train_queries, train_answers = read_and_process_dataset(training_dataset_path)
total_questions_valid, _, _, valid_passages, valid_queries, valid_answers = read_and_process_dataset(validation_dataset_path)

# Enhance answers with 'answer_end'
enhance_answers_with_end(train_answers, train_passages)
enhance_answers_with_end(valid_answers, valid_passages)

def shorten_and_process_contexts(context_list, answer_list, question_list, token_processor, length_limit, document_stride):
    shortened_contexts = []
    for index, context in enumerate(context_list):
        if len(context) > length_limit:
            answer_start_index = answer_list[index]['answer_start']
            answer_end_index = answer_start_index + len(answer_list[index]['text'])
            answer_mid_point = (answer_start_index + answer_end_index) // 2
            start_index = max(0, min(answer_mid_point - length_limit // 2, len(context) - length_limit))
            end_index = start_index + length_limit
            start_index = int(start_index)
            end_index = int(end_index)
            shortened_context = context[start_index:end_index]
            shortened_contexts.append(shortened_context)
            answer_list[index]['answer_start'] = answer_start_index - start_index
        else:
            shortened_contexts.append(context)

    processed_encodings = token_processor(
        question_list,
        shortened_contexts,
        max_length=length_limit,
        truncation=True,
        stride=document_stride,
        padding='max_length',
        return_tensors='pt'
    )
    return processed_encodings

training_encodings = shorten_and_process_contexts(training_contexts, training_answers, training_questions, token_processor, MAX_LENGTH, doc_stride)
validation_encodings = token_processor(validation_questions, validation_contexts, max_length=MAX_LENGTH, truncation=True, stride=doc_stride, padding='max_length', return_tensors='pt')


# Helper function to locate answer start and end tokens
def find_answer_positions(index):
    start_pos = 0
    end_pos = 0
    answer_encoding = token_processor(training_answers[index]['text'],  max_length=MAX_LENGTH, truncation=True, padding=True)
    for position in range(len(training_encodings['input_ids'][index]) - len(answer_encoding['input_ids'])):
        is_match = True
        for offset in range(1, len(answer_encoding['input_ids']) - 1):
            if (answer_encoding['input_ids'][offset] != training_encodings['input_ids'][index][position + offset]):
                is_match = False
                break
        if is_match:
            start_pos = position + 1
            end_pos = position + offset + 1
            break
    return (start_pos, end_pos)

start_positions = []
end_positions = []
miss_count = 0
for idx in range(len(training_encodings['input_ids'])):
    start, end = find_answer_positions(idx)
    start_positions.append(start)
    end_positions.append(end)
    if start == 0:
        miss_count += 1
    
training_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
validation_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


# Custom Dataset class for handling encodings
class EncodedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = {key: torch.tensor(value) for key, value in encodings.items()}
    
    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = EncodedDataset(training_encodings)
valid_dataset = EncodedDataset(validation_encodings)

# DataLoader creation for batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)


# Loading the BERT model
bert_transformer = BertModel.from_pretrained('path_to_model')

import torch
import torch.nn as nn
from transformers import BertModel

class AnswerPredictor(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model)
        self.dropout_layer = nn.Dropout(0.1)
        self.dense_layer1 = nn.Linear(768 * 2, 768 * 2)
        self.dense_layer2 = nn.Linear(768 * 2, 2)
        self.activation_func = nn.LeakyReLU()

    def forward(self, ids, mask, segment_ids):
        encoder_outputs = self.encoder(ids, attention_mask=mask, token_type_ids=segment_ids, return_dict=True, output_hidden_states=True)
        concat_hidden_states = torch.cat((encoder_outputs.hidden_states[-1], encoder_outputs.hidden_states[-3]), dim=-1)
        x = self.dropout_layer(concat_hidden_states)
        x = self.activation_func(self.dense_layer1(x))
        prediction_logits = self.dense_layer2(x)
        start_prediction_logits, end_prediction_logits = prediction_logits.split(1, dim=-1)
        start_prediction_logits = start_prediction_logits.squeeze(-1)
        end_prediction_logits = end_prediction_logits.squeeze(-1)

        return start_prediction_logits, end_prediction_logits

answer_predictor_model = AnswerPredictor(pretrained_model='bert-base-uncased')


# In[14]:


import torch.nn.functional as F
def calculate_focal_loss(start_predictions, end_predictions, true_start_positions, true_end_positions, focal_gamma=2.0):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_start = cross_entropy_loss(start_predictions, true_start_positions)
    loss_end = cross_entropy_loss(end_predictions, true_end_positions)
    
    start_prediction_probs = F.softmax(start_predictions, dim=1).gather(1, true_start_positions.unsqueeze(1)).squeeze(1)
    end_prediction_probs = F.softmax(end_predictions, dim=1).gather(1, true_end_positions.unsqueeze(1)).squeeze(1)
    
    focal_loss_start = ((1 - start_prediction_probs) ** focal_gamma) * loss_start
    focal_loss_end = ((1 - end_prediction_probs) ** focal_gamma) * loss_end
    
    combined_focal_loss = (focal_loss_start + focal_loss_end) / 2.0
    
    return combined_focal_loss.mean()


# In[15]:


from transformers import AdamW
optimizer = AdamW(answer_predictor_model.parameters(), lr=2e-5, weight_decay=2e-2)
scheduler = ExponentialLR(optim, gamma=0.9)
overall_accuracy = []
overall_loss = []


# In[16]:


from tqdm import tqdm

def execute_training_epoch(predictor_model, data_loader, training_epoch):
    predictor_model.train()
    epoch_losses = []
    epoch_accuracy = []
    batch_count = 0

    for batch_data in tqdm(data_loader, desc='Epoch ' + str(training_epoch) + ' Progress:'):
        optimizer.zero_grad()
        batch_ids = batch_data['input_ids'].to(device)
        batch_mask = batch_data['attention_mask'].to(device)
        batch_segment_ids = batch_data['token_type_ids'].to(device)
        batch_start_positions = batch_data['start_positions'].to(device)
        batch_end_positions = batch_data['end_positions'].to(device)
        
        predictions_start, predictions_end = predictor_model(ids=batch_ids, mask=batch_mask, segment_ids=batch_segment_ids)

        batch_loss = calculate_focal_loss(predictions_start, predictions_end, batch_start_positions, batch_end_positions, 1)  # focal_gamma=1
        epoch_losses.append(batch_loss.item())
        
        batch_loss.backward()
        optimizer.step()
        
        predictions_start_max = torch.argmax(predictions_start, dim=1)
        predictions_end_max = torch.argmax(predictions_end, dim=1)
        batch_accuracy = ((predictions_start_max == batch_start_positions).float().mean() + (predictions_end_max == batch_end_positions).float().mean()) / 2.0
        epoch_accuracy.append(batch_accuracy.item())

        if batch_count == 250 and training_epoch == 1:
            print(f'Intermediate Loss after 250 batches in epoch 1: {sum(epoch_losses)/len(epoch_losses)}')
            print(f'Intermediate Accuracy after 250 batches in epoch 1: {sum(epoch_accuracy)/len(epoch_accuracy)}')
            batch_count = 0
        
        batch_count += 1
    scheduler.step()
    average_loss = sum(epoch_losses) / len(epoch_losses)
    average_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)

    return average_accuracy, average_loss

def evaluate_model(model_to_evaluate, data_loader):
    model_to_evaluate.eval()
    answers_collected = []
    with torch.no_grad():
        for data_batch in tqdm(data_loader, desc='Evaluation in Progress'):
            ids = data_batch['input_ids'].to(device)
            masks = data_batch['attention_mask'].to(device)
            type_ids = data_batch['token_type_ids'].to(device)
            start_truth = data_batch['start_positions'].to(device)
            end_truth = data_batch['end_positions'].to(device)

            output_start, output_end = model_to_evaluate(ids, masks, type_ids)

            for j in range(ids.shape[0]):
                start_prediction = torch.argmax(output_start[j]).item()
                end_prediction = torch.argmax(output_end[j]).item() + 1
                predicted_text = tokenizerFast.decode(ids[j][start_prediction:end_prediction], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                true_text = tokenizerFast.decode(ids[j][start_truth[j]:end_truth[j]+1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                answers_collected.append((predicted_text, true_text))

    return answers_collected

import evaluate
from evaluate import load

word_error_rate = load("wer")
num_epochs = 4
model.to(device)
wer_scores = []

for epoch_idx in range(num_epochs):
    accuracy, loss = train_epoch(model, train_data_loader, epoch_idx+1)
    print(f'Epoch - {epoch_idx+1}')
    print(f'Accuracy: {accuracy}')
    print(f'Loss: {loss}')

    collected_answers = evaluate_model(model, valid_data_loader)

    predictions = [answer[0] if len(answer[0]) > 0 else "$" for answer in collected_answers]
    truths = [answer[1] if len(answer[1]) > 0 else "$" for answer in collected_answers]

    wer_result = word_error_rate.compute(predictions=predictions, references=truths)
    wer_scores.append(wer_result)

print('WER Scores for Each Epoch: ', wer_scores)
