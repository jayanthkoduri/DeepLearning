import sys
import torch
from torch.utils.data import DataLoader
import pickle
import json
from Train import Models, TestingData, evaluate_model, Attention, DecoderLSTM, EncoderLSTM
from bleu_eval import BLEU

seq2seq_model_loaded = torch.load('SavedModel/model.h5', map_location=torch.device('cpu'))
print('Seq2Seq Model loaded: ', seq2seq_model_loaded)

testing_dataset = TestingData(sys.argv[1])
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

with open('i2w.pickle', 'rb') as file_handle:
    index_to_word_map_loaded = pickle.load(file_handle)

generated_captions = evaluate_model(test_loader, seq2seq_model_loaded, index_to_word_map_loaded)

output_file_path = sys.argv[2]
with open(output_file_path, 'w') as output_file:
    for vid_id, cap_text in generated_captions:
        output_file.write(f'{vid_id},{cap_text}\n')

test_labels_data_loaded = json.load(open("data/testing_label.json"))

captions_result = {}

with open(output_file_path, 'r') as result_file:
    lines = result_file.readlines()
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        comma_idx = line.index(',')
        video_id = line[:comma_idx]
        caption_text = line[comma_idx + 1:]
        captions_result[video_id] = caption_text
        line_idx += 1

bleu_scores_list = []
item_idx = 0
while item_idx < len(test_labels_data_loaded):
    item = test_labels_data_loaded[item_idx]
    refs_captions = [cap.rstrip('.') for cap in item['caption']]
    if item['id'] in captions_result:  
        vid_bleu_score = BLEU(captions_result[item['id']], refs_captions, True)
        bleu_scores_list.append(vid_bleu_score)
    item_idx += 1

if bleu_scores_list:  
    average_bleu_score = sum(bleu_scores_list) / len(bleu_scores_list)
else:
    average_bleu_score = 0
print("Average BLEU score ", average_bleu_score)