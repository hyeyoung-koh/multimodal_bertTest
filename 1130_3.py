from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
#data=pd.read_csv('aihub_clip1-5200.csv',encoding='utf-8-sig')
#our_text=str(data['text_script'])
aihub_data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
aihub_text=aihub_data['text_script']
#print(aihub_text)
#print(aihub_text[0])
list_last_hidden_states=[]
#print(aihub_text[64975])
list_outputs=[]
for i in tqdm(range(0,101)): #인덱스가 0부터 64977이다.
    aihub_inputs=aihub_text[i] #이게 101개?이다.
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt') #aihub_final_inputs이 inputs이 되고,
    aihub_labels=torch.tensor([1]).unsqueeze(0) #aihub_labels가 labels가 된다.
    outputs=model(**aihub_final_inputs,labels=aihub_labels)
    list_outputs.append(outputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states)
    i+=1
    #i+=1
#output=list_last_hidden_states[0][0][0] #output타입을 바꿔야함.
#나는 output에 우리의 embedding을 넣도록 고쳐야함. #
#print(output.type) #tensor이다.
print(list_outputs)
num_labels=8
#bert=BertModel() #바꿔야함.
hidden_dropout_prob=0.1 #default가 0.1이다.
hidden_size=768 #default가 768이다.
dropout=torch.nn.Dropout(hidden_dropout_prob)
classifier=torch.nn.Linear(hidden_size,num_labels)
i+=1
#init_weights()

def forward():
    outputs= #우리의 CLS토큰
    #forward인수로는
    #(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,labels=None,
    #output_attentions=None,output_hidden_states=None,return_dict=None)
    pooled_output=outputs[1]
    pooled_output=dropout(pooled_output)
    logits=classifier(pooled_output)
    #pooled_output=output.logits
    #classifier=torch.nn.Linear(hidden_size,num_labels)
    #pooled_output=dropout(pooled_output)
    loss=None
    if labels is not None:
        if num_labels==1:
            loss_fct=MSELoss()
            loss=loss_fct(logits.view(-1),labels.view(-1))
        else:
            loss_fct=torch.nn.CrossEntropyLoss()
            loss=loss_fct(logits.view(-1,num_labels),labels.view(-1))
    if not return_dict:
        output=(logits,)+outputs[2:]
        return ((loss,)+output) if loss is not None else output

    return SequenceClassifierOutput(loss=loss,logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
