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
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
#clip1에 대해서만 해보자.
#wav파일을 써야함.
#mfcc 추출-clip1에 대해서만.
for i in range(1,2): #for i in range(1,2) #clip1에 대해서만 한 것임.
    mywavfile='D:\\0001-0400\\0001-0400\\clip_'+str(i)+'\\clip_'+str(i)+'.wav'
    y,sr=librosa.load(mywavfile,sr=None)
    mfcc_extracted=librosa.feature.mfcc(y=y,sr=sr) #원래는 괄호 안에 n_mfcc=20도 포함되어있었음.
    print(len(mfcc_extracted)) #20이다. n_mfcc값=len(mfcc_extracted)이다. 기본값이 n_mfcc=20이다.
    print(mfcc_extracted[0])
    #n_fft=sr*0.025,hop_length=sr*0.01 이라 하니까 오류남.
    #n_fft는 hz *0.025 이고 hop_length는 음성에서는 10ms를 기본으로 한다.
    print('mfcc추출:',mfcc_extracted)
    #print('mfcc_extracted의 type:',mfcc_extracted.type) #numpy ndarray이다.
    print('sr(sample rate):',sr)

#text 피처 추출
#csv파일을 써야함.
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

train_size=int(0.9*len(aihub_data))
val_size=len(aihub_data)-train_size
print(train_size) #90
print(val_size) #10

#divide dataset by randomly selecting samples
train_dataset,val_dataset=random_split(aihub_data,[train_size,val_size])
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

list_last_hidden_states=[]
import pandas as pd
for i in range(1,2):
    mytextfile='D:\\0001-0400\\0001-0400\\clip_'+str(i)+'\\clip_'+str(i)+'.csv'
    #mytextfile='D:\0001-0400\0001-0400\clip_1\clip_1_final.csv'
    mytext=pd.read_csv(mytextfile,encoding='cp949')
    aihub_inputs=mytext.loc[:,'text_script']
    print(len(aihub_inputs)) #10이다. 문장이 10개이므로
    print(type(aihub_inputs)) #Series이다.
    aihub_final_inputs=tokenizer(aihub_inputs[0],return_tensors='pt') #이게 text의 inputs이다.
    outputs = model(**aihub_final_inputs)
    print(outputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states) #우리의 cls 토큰은 list_last_hidden_states[0][0][0]이다.
    print('i:',i)
    print('text 피처추출벡터:',list_last_hidden_states[i-1][0][0]) #text feature추출한 벡터=>list_last_hidden_states[0][0][0]이다.원래는 [0][0][0]인데 바꿈.
    num_labels=8
    hidden_size=6768 #아까 padding시 ,뒤의 값이 6000이므로
    dropout=torch.nn.Dropout()
    classifier=torch.nn.Linear(hidden_size,8)
    loss_fct=torch.nn.CrossEntropyLoss()
    mfcc_extracted=torch.from_numpy(mfcc_extracted).float()
    print('추출한 mfcc:',mfcc_extracted)
    print('mfcc_extracted의 shape:',mfcc_extracted.shape) #[20,1547]
    print(list_last_hidden_states[i-1][0][0].shape) #출력하면 torch.Size([768])이다.<-얘는 변함없음. #원래는 [0][0][0]인데 바꿈.
