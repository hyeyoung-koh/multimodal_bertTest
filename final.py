from transformers import BertForSequenceClassification,BertTokenizerFast,Trainer,TrainingArguments,BertTokenizer
from nlp import load_dataset
import torch
import numpy as np

dataset=load_dataset('csv',data_files='clip_9csvfile_final.csv',split='train')
type(dataset) #출력결과는 nlp.arrow_dataset.Dataset이다.

#데이터를 학습/테스트셋으로 분할한다.
dataset=dataset.train_test_split(test_size=0.3)
#학습 및 테스트셋을 만든다.
train_set=dataset['train']
test_set=dataset['test']

#사전학습된 bert모델을 다운로드하자.
model=BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')
tokenizer=BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')

#한 문장이면
#tokenizer('I love Paris')
#여러 문장하려면
#tokenizer(['I love Paris','birds fly','snow fall'],padding=True,max_length=5)

#즉, 위와 같이 토크나이저를 사용해 데이터셋을 쉽게 전처리하기 위해 preprocess라는 함수를 정의해보자.
def preprocess(data):
    return tokenizer(data['text_script'],padding=True,truncation=True)
#이 preprocess함수를 사용해 학습 및 테스트셋을 전처리한다.
train_set=train_set.map(preprocess,batched=True,batch_size=len(train_set))
test_set=test_set.map(preprocess,batched=True,batch_size=len(test_set))

#set_format함수를 사용해 데이터셋에 필요한 열과 필요한 형식을 입력한다.
train_set.set_format('torch',columns=['input_ids','attention_mask','multimodal_emotion'])
test_set.set_format('torch',columns=['input_ids','attention_mask','multimodal_emotion'])

#이제 모델을 학습해보자.
#배치 및 에폭 크기를 정의한다.
batch_size=4
epochs=4
warmup_steps=500
weight_decay=0.01
#학습 인수 정의
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir='./logs',
)

#트레이너를 정의한다.
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set
)
#학습 시작
trainer.train()

#모델 평가
trainer.evaluate()