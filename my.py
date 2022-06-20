from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

#data=pd.read_csv('aihub_clip1-5200.csv',encoding='utf-8-sig')

#our_text=str(data['text_script'])
aihub_data=pd.read_csv('aihub_clip1-5200.csv',encoding='utf-8-sig')
aihub_text=aihub_data['text_script']
#print(aihub_text)
#print(aihub_text[0])
list_last_hidden_states=[]
#print(aihub_text[64975])
for i in range(0,5):
    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt')
    outputs=model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states)
    i+=1
#inputs=tokenizer(aihub_text,return_tensors='pt')
#inputs = tokenizer('나는 학교에 간다', return_tensors="pt")
#outputs = model(**inputs)
print(list_last_hidden_states) #이러면 쭉 저장됨.
print(len(list_last_hidden_states)) #10이라 출력됨.
#책:last_hidden[0][0]은 첫번째 토큰인 [CLS]의 표현 벡터를 반환한다. 
print(list_last_hidden_states[0][0]) #->이게 엑셀 1번째 text에 대한 벡터임.
#엑셀 1번째 text에 대한 CLS토큰을 구해야함.
print(list_last_hidden_states[0][0][0]) #이게 1번째 text에 대한 CLS토큰 같음.
print(list_last_hidden_states[0][0][1]) #이게 1번째 text에 대한 2번째 토큰 같음.
print(list_last_hidden_states[0][0][2]) #이게 1번째 text에 대한 3번째 토큰 같음.

print(list_last_hidden_states[1][0]) #이게 2번째 text에 대한 벡터이다.
print(list_last_hidden_states[1][0][0]) #이게 2번째 text에 대한 CLS토큰 같다.->확인완료
print(list_last_hidden_states[4][0][0]) #텍스트가 5개이므로 가장 첫번째 인덱스는 [4]이다. 즉, 얘는 4번쨰 텍스트에 대한 cls 토큰이다
print(list_last_hidden_states[1][0][1]) #이게 2번째 text에 대한 2번째 토큰벡터이다.



print(list_last_hidden_states[0][0].shape) #22,768이다.
print(list_last_hidden_states[0][0][0].shape) #768이다.
print(list_last_hidden_states[0][0].shape) #22,768이다. #18
print(list_last_hidden_states[1].shape) #1,19,768이다. #13
print(list_last_hidden_states[2].shape) #1,22,768 #
print(list_last_hidden_states[3].shape) #1,13,768  #10
print(list_last_hidden_states[6].shape) #1,94,768 이다.
#print(list_last_hidden_states[0][1].shape) #이러면 오류남.
print(list_last_hidden_states[1][0].shape)  #19,768이다.
print(list_last_hidden_states[1][0][0].shape)  #768이다.
print(list_last_hidden_states[2][0].shape)  #22,768이다.
print(list_last_hidden_states[2][0][0].shape) #768이다.
type(list_last_hidden_states[1][0][0]) #type이 torch.Tensor이다.
tolist1=list_last_hidden_states[1][0][0].tolist()
print(tolist1)
print(len(tolist1)) #768이다.

#last_hidden_states = outputs.last_hidden_state
#print(last_hidden_states)

#last_hidden_states.to('cuda') #tensor
#last_hidden_states_np=last_hidden_states.detach().numpy() #numpy
#np.save('C:\\Users\\hyeyoung\\Documents\\clip_json_aihub\\bert_feature.npy',last_hidden_states_np)

#np_load = np.load('C:\\Users\\hyeyoung\\Documents\\clip_json_aihub\\bert_feature.npy')
#print(np_load)
#print(last_hidden_states.shape)
#len(our_text)
#print(len(list_last_hidden_states))
#print(last_hidden_states[0][0])