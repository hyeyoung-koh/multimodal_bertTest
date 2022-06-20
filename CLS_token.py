from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',num_labels=8,output_attentions=False,output_hidden_states=False)

#data=pd.read_csv('aihub_clip1-5200.csv',encoding='utf-8-sig')
#our_text=str(data['text_script'])
aihub_data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
aihub_text=aihub_data['text_script']
#print(aihub_text)
#print(aihub_text[0])
list_last_hidden_states=[]
len(aihub_data) #100이다.

for i in tqdm(range(0,99)): #인덱스가 0부터 64977이다.
    print(i)
    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt')
    #outputs=model(**aihub_final_inputs)
    outputs = model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states)
#inputs=tokenizer(aihub_text,return_tensors='pt')
#inputs = tokenizer('나는 학교에 간다', return_tensors="pt")
#outputs = model(**inputs)
print(list_last_hidden_states) #이러면 쭉 저장됨.
print(list_last_hidden_states[0])
print(list_last_hidden_states[0][0]) #이게 1번째 문장.즉,i=1
print(list_last_hidden_states[1][0]) #이게 2번쨰 문장.즉,i=2
print(list_last_hidden_states[0][0][0])
print(list_last_hidden_states[0][0][1])
print(list_last_hidden_states[0][0][2])
#========<2번째 문장>===================
print(list_last_hidden_states[1][0][0]) #이게 2번째 문장의 cls 토큰
print(list_last_hidden_states[1][0][1]) #이게 2번째 문장의 2번째 요소의 임베딩

#===============================================
print(list_last_hidden_states[0])
print(list_last_hidden_states[0][0][0])
print(list_last_hidden_states[1][0][0])
#class torch.nn.Dropout(p=0.5,inplace=False)
#aihub_data.iloc[:,9]
num_labels=8
hidden_size=768
#classifier_dropout=classifier_dropout()
dropout=torch.nn.Dropout()
classifier=torch.nn.Linear(hidden_size,8)
#labels=aihub_data.iloc[:,10]
#pooled_output=list_last_hidden_states[0][0][0]
#loss_fct=torch.nn.BCEWithLogitsLoss()
loss_fct=torch.nn.CrossEntropyLoss()
#loss_fct=torch.nn.MultiLabelSoftMarginLoss()
for i in tqdm(range(0,100)):
    pooled_output=list_last_hidden_states[i][0][0]
    print('pooled_output:',pooled_output)
    print('pooled_output의 shape:',pooled_output.shape)
    pooled_output=dropout(pooled_output)

    #labels=torch.tensor(aihub_data.iloc[i,10]).unsqueeze(-1) #인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
    labels = torch.tensor(aihub_data.iloc[i, 11])

    print('original labels:',labels) #labels출력하면 tensor([1])과 같은 형태로 출력됨
    #labels=torch.tensor(labels)
    #labels=torch.tensor([labels]).unsqueeze(1)
    #labels=torch.tensor(labels)
    logits=classifier(pooled_output)
    logits=logits.reshape(1,8)#추가한 코드
    labels=labels.reshape(1)
    #print(logits)
    #print(labels.shape,logits.shape)
    #loss_fct=torch.nn.BCEWithLogitsLoss()
    #loss=loss_fct(logits,labels) #여기서 오류 발생
    loss=loss_fct(logits,labels)
    print('labels-reshape:',labels)
    print('logits:',logits)
    print('loss:',loss) 

# for i in range(0,101):
#     labels=aihub_data.iloc[i,10]
#     print(labels)
#     print(type(labels))
#print(aihub_data.iloc[1,10])
#print(type(aihub_data.iloc[1,10])) #numpy.int64

