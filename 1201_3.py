# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import pandas as pd
# import tqdm
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
# aihub_data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
# aihub_text=aihub_data['text_script']
# # #print(aihub_text)
# # #print(aihub_text[0])
# # list_last_hidden_states=[]
# # #print(aihub_text[64975])
# # list_outputs=[]
# for i in range(0,101): #인덱스가 0부터 64977이다.
#     aihub_inputs=aihub_text[i] #이게 101개?이다.
#     #aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt') #aihub_final_inputs이 inputs이 되고,
#     aihub_final_inputs=tokenizer(aihub_inputs,padding="max_length",truncation=True)
#     model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=8,
#                                                           output_attentions=False, output_hidden_states=False)
#outputs의 형태는 self.bert의 결과이다.
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',num_labels=8,output_attentions=False,output_hidden_states=False)
# model = BertModel.from_pretrained('bert-base-multilingual-uncased')
#data=pd.read_csv('aihub_clip1-5200.csv',encoding='utf-8-sig')
#our_text=str(data['text_script'])
aihub_data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
aihub_text=aihub_data['text_script']
# #print(aihub_text)
# #print(aihub_text[0])
# list_last_hidden_states=[]
# #print(aihub_text[64975])
# list_outputs=[]
for i in range(0,101): #인덱스가 0부터 64977이다.
    # aihub_inputs=aihub_text[i] #이게 101개?이다.
    # aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt') #aihub_final_inputs이 inputs이 되고,
    # aihub_labels=torch.tensor([1]).unsqueeze(0) #aihub_labels가 labels가 된다.
    # outputs=model(**aihub_final_inputs,labels=aihub_labels)
    # loss=outputs.loss
    # print(loss)
    # logits=outputs.logits
    # print(logits)
    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt')
    aihub_labels=torch.tensor([1]).unsqueeze(0)
    outputs=model(**aihub_final_inputs,labels=aihub_labels)
    #pooled_output=outpus
    print()
    i+=1


pooled_output=#우리의 cls토큰
pooled_output=dropout(pooled_output)
logits=classifier(pooled_output)
loss_fct=BCEWithLogitsLoss()
loss=loss_fct(logits,labels)

