from transformers import BertTokenizer, BertModel
import pandas as pd
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
text=data['text_script']
list_last_hidden_states=[]
#for i in range(0,100):
#     inputs=text[i]
#     final_inputs=tokenizer(inputs,return_tensors='pt')
#     outputs=model(**final_inputs)
#     last_hidden_states=outputs.last_hidden_state
#     list_last_hidden_states.append(last_hidden_states)
#     i+=1
# print('100개 문장에 대한 임베딩 벡터 리스트:',list_last_hidden_states)
for i in range(0,1):
    inputs=text[i]
    final_inputs=tokenizer(inputs,return_tensors='pt')
    outputs=model(**final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states)
    i+=1
print('1번째 문장:',data['text_script'][0])
print('1번째 문장 임베딩 벡터:',list_last_hidden_states[0])
print('1번째 문장 임베딩 벡터 shape:',list_last_hidden_states[0].shape)