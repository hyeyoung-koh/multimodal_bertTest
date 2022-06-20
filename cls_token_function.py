from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
aihub_data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
aihub_text=aihub_data['text_script']

list_last_hidden_states=[]
#print(aihub_text[64975])
for i in tqdm(range(0,100)): #인덱스가 0부터 64977이다.
    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt')
    #outputs=model(**aihub_final_inputs)
    outputs = model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states)
    i+=1

num_labels=8
hidden_size=768
dropout=torch.nn.Dropout()
classifier=torch.nn.Linear(hidden_size,8)
loss_fct=torch.nn.CrossEntropyLoss() #이거 확인해야함.

def classifier(i):
    pooled_output=list_last_hidden_states[i][0][0]
    labels=torch.tensor(aihub_data.iloc[i,11])
    logits = classifier(pooled_output)
    logits=logits.reshape(1,8)
    loss=loss_fct(logits,labels)
    print('labels-reshape:',labels)
    print('logits:',logits)
    print('loss:',loss)



