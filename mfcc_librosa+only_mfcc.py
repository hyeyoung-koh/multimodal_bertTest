import scipy
import scipy.signal
import scipy.fftpack
import librosa
import sklearn
import librosa.display
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
aihub_data=pd.read_csv('clip10_sample.csv',encoding='utf-8-sig')
print(len(aihub_data)) #115개
aihub_text=aihub_data['text_script']
list_last_hidden_states=[]

for i in range(1,11): # i=1부터 10까지
    mycsv = pd.read_csv('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip_' + str(i) + 'myfinal.csv', encoding='utf-8-sig')
    for j in range(0,len(mycsv)):#0부터 len(mycsv)-1까지
    #C:\Users\hyeyoung\PycharmProjects\test\melspectrogram
        myfile='C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip'+str(i)+'_'+str(j)+'_cut.wav'
        #myfile='D:\\0001-0400\\0001-0400\\clip_'+str(i)+'\\clip_'+str(i)+'.wav'
        y,sr=librosa.load(myfile,sr=None)
        mfcc_extracted=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
        print('myfile명:',myfile)
        #print('mfcc추출:',mfcc_extracted)
        #print('sr(sample rate):',sr)
        mfcc_extracted=torch.from_numpy(mfcc_extracted).float()
        print('추출한 mfcc:',mfcc_extracted)
        print('mfcc_extracted의 shape:',mfcc_extracted.shape)
        print('mfcc_extracted.size(0):', mfcc_extracted.size(0))
        mfcc_extracted_reshape=mfcc_extracted.view(mfcc_extracted.size(0),-1)
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
        print(librosa.display.specshow(padded_mfcc, x_axis='time'))
        print('mfcc_extracted_reshape의 shape:',mfcc_extracted_reshape.shape)

for i in range(len(aihub_data)):
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



    pooled_output=torch.cat([list_last_hidden_states_reshape,padded_mfcc],dim=1) #list_last_hidden_states_reshape+padded_mfcc
        #concat 텐서 합치기
        #h1=tf.concat([list_last_hidden_states[i][0][0],mfcc_extracted],0) #0:행을 기준으로 합침, 1:열을 기준으로 합침
        #pooled_output=tf.concat([list_last_hidden_states[i][0][0],mfcc_extracted],1)
        #stack함수를 이용한다면
        #pooled_output=torch.stack([list_last_hidden_states[i][0][0],mfcc_extracted_flatten],dim=1) #stack은 세로로 쌓는다.
        #pooled_output=torch.stack([list_last_hidden_states[i][0][0],mfcc_extracted_flatten],dim=0)#만약 행 기준으로 합치는 거라면
        #pooled_output=torch.cat((list_last_hidden_states[i][0][0],mfcc_extracted),1) #->이거 오류남. #dim=1로 concat하려면 각 텐서의 행의 수가 같아야한다.
    print('pooled_output:',pooled_output)
    print('pooled_output의 shape:',pooled_output.shape) #141739+768=142507이다. 이거 출력하면 torch.Size([142507])이라 출력됨.
    pooled_output=dropout(pooled_output)
    #labels=torch.tensor(aihub_data.iloc[i,10]).unsqueeze(-1) #인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
    labels = torch.tensor(aihub_data.iloc[i, 11]) #얘가 찐 정답
    print('original labels:',labels) #labels출력하면 tensor([1])과 같은 형태로 출력됨
        #labels=torch.tensor(labels)
        #labels=torch.tensor([labels]).unsqueeze(1)
        #labels=torch.tensor(labels)
    logits=classifier(pooled_output) #여기서 multiply하는 과정에서 오류 발생 #(bs,dim) #pooled_output([cls]토큰을 분류 계층에 전파 )
        #float type의 객체가 들어올 줄 알았는데 double type의 객체가 들어왔다는 에러이다.
        #이를 위해 tensor로 바꿔주고 나서, .float()를 넣어서 float자료형으로 변환해주어야한다.
        #ex)tran_data=torch.from_numpy(np_train_data).float()
        #logits=classifier(pooled_output.unsqueeze(-1))
        #logits:(batch_size,config.num_labels)가 shape이다. 그래서 768*8이다.
        #logits=logits.reshape(1,8)#추가한 코드
        #reshaped_logits=logits.view(-1,8)
    labels=labels.reshape(1)
        #print(logits)
        #print(labels.shape,logits.shape)
        #loss_fct=torch.nn.BCEWithLogitsLoss()
        #loss=loss_fct(logits,labels) #여기서 오류 발생
    loss=loss_fct(logits,labels)
        #outputs=(logits,)+discriminator_hidden_states[1:] #임시 예시로.
    print('labels-reshape:',labels)
    print('logits:',logits)
    print('loss:',loss)
