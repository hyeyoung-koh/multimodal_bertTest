from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import sklearn
import librosa
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

def mfcc_feature(myfile):
    y, sr = librosa.load(myfile, sr=None)
    mfcc_extracted = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print('myfile명:', myfile)
    mfcc_extracted = torch.from_numpy(mfcc_extracted).float()
    print('추출한 mfcc:', mfcc_extracted)
    print('mfcc_extracted의 shape:', mfcc_extracted.shape)
    print('mfcc_extracted.size(0):', mfcc_extracted.size(0))
    mfcc_extracted_reshape = mfcc_extracted.view(mfcc_extracted.size(0), -1)
    mfcc_extracted_reshape = mfcc_extracted.view(1, -1)
    print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)
    # mfcc scaling
    mfcc_scale = sklearn.preprocessing.scale(mfcc_extracted_reshape, axis=1)
    print('mfcc_scale:', mfcc_scale)
    print('mfcc_scale의 shape:', mfcc_scale.shape)
    pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    padded_mfcc = pad2d(mfcc_scale, 6000)
    print('padded_mfcc:', padded_mfcc)
    print('padded_mfcc의 shape:', padded_mfcc.shape)
    print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)
    padded_mfcc = torch.Tensor(padded_mfcc)
    return padded_mfcc

def text_feature(aihub_inputs):
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt')
    outputs = model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    print('last_hidden_states[0][0]:',last_hidden_states[0][0])
    print('last_hidden_states의 shape:',last_hidden_states.shape)
    cls_token=last_hidden_states[0][0]
    cls_token_final=last_hidden_states[0][0].reshape(1,-1)
    return cls_token_final

# for i in range(1,10):
#     mycsv = pd.read_csv('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip_' + str(i) + 'myfinal.csv', encoding='utf-8-sig')
#     print(i)
#     for j in range(0,len(mycsv)):
#         myfile = 'C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip' + str(i) + '_' + str(j) + '_cut.wav'
#         mfcc_padded_feature=mfcc_feature(myfile)
#         print(type(mfcc_padded_feature))
#         print('j:',j)
#         text_feature_extract=text_feature(mycsv.loc[j,'text_script'])
#         num_labels = 8
#         hidden_size = 6768
#         dropout = torch.nn.Dropout()
#         classifier = torch.nn.Linear(hidden_size, 8)
#         loss_fct = torch.nn.CrossEntropyLoss()
#         pooled_output = torch.cat([mfcc_padded_feature, text_feature_extract], dim=1)
#         print('pooled_output:', pooled_output)
#         print('pooled_output의 shape:', pooled_output.shape)
#         pooled_output = dropout(pooled_output)
#         labels = torch.tensor(mycsv.loc[j, 'emotion_num'])
#         print('original labels:', labels)
#         logits = classifier(pooled_output)
#         labels = labels.reshape(1)
#         loss = loss_fct(logits, labels)
#         print('labels-reshape:', labels)
#         print('logits:', logits)
#         print('loss:', loss)
