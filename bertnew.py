# from transformers import BertTokenizer, BertModel
# import pandas as pd
# import torch
# import numpy as np
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained('bert-base-multilingual-cased')
#
# inputs=tokenizer('나는 학교에 간다.',return_tensors='pt')
# #inputs=tokenizer('[CLS] 한국어 모델을 공유합니다. [SEP]')
#
# outputs=model(**inputs)
# last_hidden_states=outputs.last_hidden_state
# print(last_hidden_states)
# print(last_hidden_states.shape)

# from transformers import BertModel,BertTokenizer
# import torch
#
# model=BertModel.from_pretrained('bert-base-multilingual-uncased')
# tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


# from transformers.modeling_bert import BertModel, BertForMaskedLM
# from tokenization_kobert import KoBertTokenizer
#
# tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
# model = BertModel.from_pretrained("monologg/kobert")
#
# #model = BertModel.from_pretrained("monologg/kobert")
# sentence='나는 학교에 간다.'
#
# tokens=tokenizer.tokenize(sentence)
# print(tokens)
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# print(outputs)

from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
#model=BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=8,output_attentions=False,output_hidden_states=False)
inputs = tokenizer("나는 학교에 간다.", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1 #labels를 출력하면 tensor([[1]])이라 출력된다.
#unsqueeze(0)이므로 1번째 차원에 1인 차원을 추가한다는 뜻이다.
#unsqueeze(1)이면 2번째 차원에 1을 추가하겠다는 뜻이다.
#unsqueeze(-1)이면 마지막 차원에 1을 추가하겠다는 뜻이다.
print(labels) #labels를 출력하면 tensor([[1]]) 이라고 출력된다.
outputs = model(**inputs, labels=labels)
print(outputs)
