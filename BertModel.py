from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

inputs=tokenizer('나는 학교에 간다.',return_tensors='pt')
#inputs=tokenizer('[CLS] 한국어 모델을 공유합니다. [SEP]')

outputs=model(**inputs)
last_hidden_states=outputs.last_hidden_state
print(last_hidden_states)
print(last_hidden_states.shape) #1,11,768
print(last_hidden_states[0][0][0]) #1,11,768 보험 하나만 들어줄래?
print(last_hidden_states[0][0]) #이게 CLS 토큰의 임베딩 벡터이다.
print(last_hidden_states[0][0].shape) #이걸 출력하면 torch.Size([768])이다.

print(last_hidden_states[0][0][1])
print(last_hidden_states[0][11])

print(last_hidden_states.shape) #1,13,768
print(last_hidden_states[0].shape) #13,768
print(last_hidden_states[0][0].shape) #768

#print(last_hidden_states[0][0])