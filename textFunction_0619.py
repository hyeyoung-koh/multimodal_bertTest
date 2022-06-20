from transformers import BertTokenizer, BertModel
import pandas as pd
import torch


def text_feature(aihub_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    print(len(aihub_data))  # 115개
    aihub_text = aihub_data['text_script']
    list_last_hidden_states = []
    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt') #이게 text의 inputs이다.
    #outputs=model(**aihub_final_inputs)
    outputs = model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states) #우리의 cls 토큰은 list_last_hidden_states[0][0][0]이다.
    print('i:',i)
    print('text 피처추출벡터:',list_last_hidden_states[i-1][0][0]) #text feature추출한 벡터=>list_last_hidden_states[0][0][0]이다.원래는 [0][0][0]인데 바꿈.
    num_labels=8
    hidden_size=6768 #아까 padding시 ,뒤의 값이 6000이므로
    dropout=torch.nn.Dropout()
    classifier=torch.nn.Linear(hidden_size,8)
    loss_fct=torch.nn.CrossEntropyLoss()
    print(list_last_hidden_states[i - 1][0][0].shape)  # 출력하면 torch.Size([768])이다.<-얘는 변함없음. #원래는 [0][0][0]인데 바꿈.
    hidden_states=list_last_hidden_states[i-1][0][0] #원래 [0][0][0]인데 바꿈.
    #list_last_hidden_states_reshape =hidden_states.view(hidden_states.size(0),-1)
    list_last_hidden_states_reshape = hidden_states.view(-1,hidden_states.size(0))
    #list_last_hidden_states_reshape = hidden_states.view(-1,hidden_states.size(0))
    print('hidden_states의 shape:',hidden_states.shape) #[768]
    print('list_last_hidden_states_reshape의 shape:',list_last_hidden_states_reshape.shape) #이러면 size가 [1,768]이다.


aihub_data=pd.read_csv('clip10_sample.csv', encoding='utf-8-sig')
for i in range(len(aihub_data)):
    text_feature(aihub_data)
