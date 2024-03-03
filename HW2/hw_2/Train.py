import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import random
from scipy.special import expit

import sys
import os
import json
import re
import pickle
import time

from collections import Counter
import json
import re

def preprocess_text_data():
    file_path = 'data/'

    word_frequency = Counter()

    with open(file_path + 'training_label.json', 'r') as file_handle:
        data = json.load(file_handle)

    
    for entry in data:
        for sentence in entry['caption']:
            words_in_sentence = re.sub('[.!,;?]', ' ', sentence).lower().split()
            word_frequency.update(words_in_sentence)

    filtered_word_dict = {word: freq for word, freq in word_frequency.items() if freq > 4}
    special_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    index_to_word = {index + len(special_tokens): word for index, word in enumerate(filtered_word_dict)}
    word_to_index = {word: index + len(special_tokens) for index, word in enumerate(filtered_word_dict)}
    for token, index in special_tokens:
        index_to_word[index] = token
        word_to_index[token] = index

    return index_to_word, word_to_index, filtered_word_dict

def encode_sentence(input_text, vocabulary, word_to_index):
    tokens = re.sub(r'[.!,;?]', ' ', input_text).split()
    encoded_text = [word_to_index.get(token, 3) for token in tokens]
    encoded_text = [1] + encoded_text + [2]
    return encoded_text

def annotate(file_name, vocab, index_map):
    full_path = f'data/{file_name}'
    processed_captions = []
    with open(full_path, 'r') as json_file:
        data_labels = json.load(json_file)

    for item in data_labels:
        for text in item['caption']:
            transformed_text = encode_sentence(text, vocab, index_map)
            processed_captions.append((item['id'], transformed_text))

    return processed_captions

def avi(directory_path):
    loaded_data = {}
    data_folder_path = 'data/' + directory_path
    numpy_files = os.listdir(data_folder_path)
    file_index = 0
    for numpy_file in numpy_files:
        print(f"Processing file number: {file_index}")
        file_index += 1
        data = np.load(os.path.join(data_folder_path, numpy_file))
        key_name = numpy_file.rsplit('.npy', 1)[0]
        loaded_data[key_name] = data
    return loaded_data

def prepare_batch(batch_data):
    batch_data.sort(key=lambda item: len(item[1]), reverse=True)
    avi_list, caption_list = zip(*batch_data)
    avi_tensor = torch.stack(avi_list, dim=0)
    caption_lengths = list(map(len, caption_list))
    max_length = max(caption_lengths)
    caption_targets = torch.zeros(len(caption_list), max_length).long()

    for index, caption in enumerate(caption_list):
        caption_end = caption_lengths[index]
        caption_targets[index, :caption_end] = torch.LongTensor(caption[:caption_end])

    return avi_tensor, caption_targets, caption_lengths

class TrainingData(Dataset):
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.w2i = w2i
        self.avi = avi(label_file)
        self.data_pair = annotate(files_dir, word_dict, w2i)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError("Index exceeds dataset size.")
        
        video_name, text = self.data_pair[index]
        video_tensor = torch.Tensor(self.avi[video_name])
        video_tensor += torch.rand(video_tensor.size()) * 0.2 
        text_tensor = torch.tensor(text, dtype=torch.long)
        
        return video_tensor, text_tensor

    
class TestingData(Dataset):
    def __init__(self, test_data_path):
        self.dataset_entries = []
        for filename in os.listdir(test_data_path):
            identifier = filename.split('.npy')[0]
            data = np.load(os.path.join(test_data_path, filename))
            self.dataset_entries.append([identifier, data])
            
    def __len__(self):
        return len(self.dataset_entries)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        return self.dataset_entries[index]

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_layer1 = nn.Linear(2*hidden_size, hidden_size)
        self.attention_layer2 = nn.Linear(hidden_size, hidden_size)
        self.attention_layer3 = nn.Linear(hidden_size, hidden_size)
        self.attention_layer4 = nn.Linear(hidden_size, hidden_size)
        self.compute_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, hidden_dim = encoder_outputs.size()
        hidden_state_expanded = hidden_state.view(batch_size, 1, hidden_dim).repeat(1, seq_len, 1)
        combined_inputs = torch.cat((encoder_outputs, hidden_state_expanded), 2).view(-1, 2*self.hidden_size)

        attn_hidden = self.attention_layer1(combined_inputs)
        attn_hidden = self.attention_layer2(attn_hidden)
        attn_hidden = self.attention_layer3(attn_hidden)
        attn_hidden = self.attention_layer4(attn_hidden)
        weights = self.compute_weight(attn_hidden)
        weights = weights.view(batch_size, seq_len)
        weights_normalized = F.softmax(weights, dim=1)
        context_vector = torch.bmm(weights_normalized.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector

    
class EncoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding_layer = nn.Linear(4096, 512)
        self.dropout_layer = nn.Dropout(0.33)
        self.lstm_layer = nn.LSTM(512, 512, batch_first=True)

    def forward(self, x):
        batch_size, sequence_length, features = x.shape    
        x = x.reshape(-1, features)
        x = self.embedding_layer(x)
        x = self.dropout_layer(x)
        x = x.reshape(batch_size, sequence_length, 512)

        lstm_output, (hidden_state, cell_state) = self.lstm_layer(x)
        return lstm_output, hidden_state

    
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.33):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(dropout_percentage)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # Renamed to fc for clarity

    def forward(self, encoder_hidden_state_last, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_hidden_state_last.size()
        decoder_hidden_state = encoder_hidden_state_last
        decoder_context = torch.zeros(decoder_hidden_state.size())#.cuda()
        decoder_input_word = Variable(torch.ones(batch_size, 1)).long()#.cuda()
        sequence_log_probabilities = []
        sequence_predictions = []

        targets_embedded = self.embedding(targets)
        _, sequence_length, _ = targets_embedded.size()

        for i in range(sequence_length - 1):
            teacher_forcing_threshold = self.calculate_teacher_forcing_ratio(tr_steps)
            if random.uniform(0.05, 0.995) > teacher_forcing_threshold:
                input_word = targets_embedded[:, i]
            else:
                input_word = self.embedding(decoder_input_word).squeeze(1)

            context_vector = self.attention(decoder_hidden_state, encoder_output)
            lstm_input = torch.cat([input_word, context_vector], dim=1).unsqueeze(1)
            lstm_output, (decoder_hidden_state, decoder_context) = self.lstm(lstm_input, (decoder_hidden_state, decoder_context))
           
            log_probability = self.fc(lstm_output.squeeze(1))
            sequence_log_probabilities.append(log_probability.unsqueeze(1))
            decoder_input_word = log_probability.unsqueeze(1).max(2)[1]

        sequence_log_probabilities = torch.cat(sequence_log_probabilities, dim=1)
        sequence_predictions = sequence_log_probabilities.max(2)[1]

        return sequence_log_probabilities, sequence_predictions

    def infer(self, encoder_hidden_state_last, encoder_output):
        _, batch_size, _ = encoder_hidden_state_last.size()
        decoder_hidden_state = encoder_hidden_state_last
        decoder_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_context = torch.zeros(decoder_hidden_state.size())
        sequence_log_probabilities = []
        sequence_predictions = []
        predicted_sequence_length = 28

        for i in range(predicted_sequence_length - 1):
            input_word = self.embedding(decoder_input_word).squeeze(1)
            context_vector = self.attention(decoder_hidden_state, encoder_output)
            lstm_input = torch.cat([input_word, context_vector], dim=1).unsqueeze(1)
            lstm_output, (decoder_hidden_state, decoder_context) = self.lstm(lstm_input, (decoder_hidden_state, decoder_context))
            
            log_probability = self.fc(lstm_output.squeeze(1))
            sequence_log_probabilities.append(log_probability.unsqueeze(1))
            decoder_input_word = log_probability.unsqueeze(1).max(2)[1]

        sequence_log_probabilities = torch.cat(sequence_log_probabilities, dim=1)
        sequence_predictions = sequence_log_probabilities.max(2)[1]

        return sequence_log_probabilities, sequence_predictions

    def calculate_teacher_forcing_ratio(self, training_steps):
        return expit(training_steps / 20 + 0.85)
    

class Models(nn.Module):
    def __init__(self, encoder, decoder):
        super(Models, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_hidden_state_last = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_hidden_state_last=encoder_hidden_state_last, encoder_output=encoder_outputs,
                                                        targets=target_sentences, mode=mode, tr_steps=tr_steps)
        elif mode == 'inference':
            # Corrected variable name from `encoder_last_hidden_state` to `encoder_hidden_state_last`
            seq_logProb, seq_predictions = self.decoder.infer(encoder_hidden_state_last=encoder_hidden_state_last, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions
    
def calculate_loss(loss_fn, predictions, targets, lengths):
    device = predictions.device
    batch_size = predictions.shape[0]
    
    # Ensure predictions are on the correct device and are of floating point type
    concatenated_predictions = torch.Tensor().to(device).float()  # This should already be the case
    
    # Ensure targets are on the correct device and are of integer type, suitable for cross_entropy
    concatenated_targets = torch.LongTensor().to(device)  # Targets must be integers
    
    for index in range(batch_size):
        sequence_length = lengths[index] - 1
        sliced_prediction = predictions[index, :sequence_length]
        sliced_target = targets[index, :sequence_length]
        concatenated_predictions = torch.cat((concatenated_predictions, sliced_prediction), dim=0)
        concatenated_targets = torch.cat((concatenated_targets, sliced_target.type(torch.long)), dim=0)  # Ensure targets are long
    
    total_loss = loss_fn(concatenated_predictions, concatenated_targets)
    avg_loss = total_loss / batch_size
    return avg_loss

def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions = zip(*data)
    video_tensor = torch.stack(videos, dim=0)
    cap_lengths = [len(cap) for cap in captions]
    cap_tensor = torch.zeros(len(captions), max(cap_lengths), dtype=torch.long)
    for idx, cap in enumerate(captions):
        length = cap_lengths[idx]
        cap_tensor[idx, :length] = torch.tensor(cap[:length], dtype=torch.long)
    return video_tensor, cap_tensor, cap_lengths

def train(model, epoch, loss_fn, parameters, optimizer, train_loader):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        # avi_feats, ground_truths = Variable(avi_feats).cuda(), Variable(ground_truths).cuda()
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)
        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)
        print('Batch - ', batch_idx, ' Loss - ', loss)
        loss.backward()
        optimizer.step()

    loss_value = loss.item()
    return loss_value

def evaluate_model(data_loader, neural_model, index_to_word):
    neural_model.eval()
    predictions_list = []

    for batch_index, data_batch in enumerate(data_loader):
        ids, features = data_batch
        features_gpu = features
        _, processed_features = ids, Variable(features_gpu).float()
        _, predicted_sequences = neural_model(processed_features, mode='inference')
        decoded_predictions = [[index_to_word[token.item()] if index_to_word[token.item()] != '<UNK>' else 'something' for token in sequence] for sequence in predicted_sequences]
        sentences = [' '.join(sequence).split('<EOS>')[0] for sequence in decoded_predictions]
        id_sentence_pairs = zip(ids, sentences)
        for pair in id_sentence_pairs:
            predictions_list.append(pair)
            
    return predictions_list

def main():

    index_to_word, word_to_index, word_dict = preprocess_text_data()
    with open('i2w.pickle', 'wb') as file:
        pickle.dump(index_to_word, file, protocol=pickle.HIGHEST_PROTOCOL)
    label_directory = 'training_data/feat'
    files_directory = 'training_label.json'
    training_data = TrainingData(label_directory, files_directory, word_dict, word_to_index)
    dataloader = DataLoader(dataset=training_data, batch_size=64, shuffle=True, num_workers=8, collate_fn=minibatch)
    
    num_epochs = 20
    dropout_rate = 0.33

    encoder = EncoderLSTM()
    decoder = DecoderLSTM(512, len(index_to_word) + 4, len(index_to_word) + 4, 1024, dropout_rate)
    model = Models(encoder=encoder, decoder=decoder)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    training_losses = []

    for epoch in range(num_epochs):
        epoch_loss = train(model, epoch + 1, criterion, model.parameters(), optimizer, dataloader)
        training_losses.append(epoch_loss)
    
    torch.save(model, "SavedModel/model.h5")
    print("Completed Training")

if __name__ == "__main__":
    main()