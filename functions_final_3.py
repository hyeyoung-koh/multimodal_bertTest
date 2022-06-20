from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import sklearn
import librosa
import torch
import random
import time

# GPU / CPU 할당 코드
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# 음성 특징 추출 (MFCC)
# @@ 프린트 된 콘솔창 캡쳐
def mfcc_feature(myfile):  # myfile: 한 문장에 대한 wav 파일
    # MFCC 추출
    y, sr = librosa.load(myfile, sr=None)
    mfcc_extracted = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print('myfile명:', myfile)
    mfcc_extracted = torch.from_numpy(mfcc_extracted).float()
    print('추출한 mfcc:', mfcc_extracted)
    print('mfcc_extracted의 shape:', mfcc_extracted.shape)
    print('mfcc_extracted.size(0):', mfcc_extracted.size(0))
    # scaling & padding
    mfcc_extracted_reshape = mfcc_extracted.view(1, -1)  # @@ 이게 맞음.ㄴ
    #mfcc_extracted_reshape = mfcc_extracted.view(mfcc_extracted.size(0), -1)  # @@
    print('mfcc_extracted_reshape:',mfcc_extracted_reshape)
    print('mfcc_extracted_reshape의 shape:',mfcc_extracted_reshape.shape)
    #print('mfcc_extracted_reshape2:',mfcc_extracted_reshape)
    #print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)

    mfcc_scale = sklearn.preprocessing.scale(mfcc_extracted_reshape, axis=1)
    print('mfcc_scale:', mfcc_scale)
    print('mfcc_scale의 shape:', mfcc_scale.shape)

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    padded_mfcc = pad2d(mfcc_scale, 6000)  # 패딩
    print('padded_mfcc:', padded_mfcc)
    print('padded_mfcc의 shape:', padded_mfcc.shape)
    print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)

    padded_mfcc = torch.Tensor(padded_mfcc)
    print(padded_mfcc)  # @@ 제가 임의로 추가했으용 ㅋ

    return padded_mfcc  # output: 한 문장에 대한 wav 파일의 MFCC 값 (패딩 적용)

mfcc_feature('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip1_0_cut.wav')

# 텍스트 특징 추출 (BERT)
# @@ 프린트 된 콘솔창 캡쳐
def text_feature(aihub_inputs):  # aihub_inputs: 한 문장에 대한 txt
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')  # @@ 함수 밖에서 선언, 매개변수 default로 넣어주기
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')  # @@ 함수 밖에서 선언, 매개변수 default로 넣어주기
    aihub_final_inputs = tokenizer(aihub_inputs, return_tensors='pt')
    outputs = model(**aihub_final_inputs)  # @@ ** 의미 찾기
    last_hidden_states = outputs.last_hidden_state
    print('last_hidden_states[0][0]:', last_hidden_states[0][0])
    print('last_hidden_states의 shape:', last_hidden_states.shape)
    cls_token = last_hidden_states[0][0]  # @@
    cls_token_final = last_hidden_states[0][0].reshape(1, -1)  # @@
    print('cls_token_final의 shape:', cls_token_final.shape)

    return cls_token_final  # output: 한 문장에 대한 cls hs 값

dataset = []  # concat vector들을 삽입하는 리스트
labels = []  # 대응되는 label들을 삽입하는 리스트

# clip10개에 대해 한 문장씩 불러오기
# clip1: 10 clip2: 10 clip3: 13 clip4: 11 clip5: 10 clip6: 12 clip7: 14 clip8: 11 clip9: 12 clip10: 12  =>총 115개 문장
for i in range(1, 10):  # i : clip num
    mycsv = pd.read_csv('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip_' + str(i) + 'myfinal.csv',
                        encoding='utf-8-sig')
    print(i)

    for j in range(0, len(mycsv)):  # j : 행 idx
        print('j:', j)

        # 한 문장에 대한 padded mfcc 추출
        myfile = 'C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip' + str(i) + '_' + str(j) + '_cut.wav'
        mfcc_padded_feature = mfcc_feature(myfile)
        print(type(mfcc_padded_feature))  # torch.tensor이다.

        # 한 문장에 대한 cls hs 추출
        text_feature_extract = text_feature(mycsv.loc[j, 'text_script'])

        # 각 vec을 np로 변환
        numpy_mfcc_feature = mfcc_padded_feature.cpu().numpy()  # @@
        print(numpy_mfcc_feature)
        numpy_text_feature = text_feature_extract.cpu().detach().numpy()  # @@
        print(numpy_text_feature)

        # numpy_mfcc_feature와 numpy_text_feature를 concat
        concat = np.concatenate((numpy_mfcc_feature, numpy_text_feature), axis=1)
        dataset.append(concat)

        # labels값 불러오기 (labels: 감정을 정수로 표현한 값들의 리스트 0-7)
        labels.append(torch.tensor(mycsv.loc[j, 'emotion_num']))

print('labels:', labels)
print(len(labels))  # 103
print(dataset)
print(len(dataset))  # len(dataset)=103이다.

# train vs validation = 9:1로 분리
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Divide the dataset by randomly selecting samples.
# dataset을 train_dataset과 val_dataset으로 분리
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # @@ 9:1 잘 분할 되었는지 확인 =>92/11

# dataloader 기반 batch 생성
batch_size = 16  # @@ 32로 늘리기

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size,  # Trains with this batch size.)
)  # 현재 batch_size=16
print(len(train_dataloader))  # @@ train data 수 / batch size 나오는 지 확인 #3이라 출력됨 .

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=batch_size,  # Evaluate with this batch size.)
)  # 현재 batch_size=16
print(len(validation_dataloader))  # @@ validation data 수 / batch size 나오는 지 확인 #1이라 출력됨.

'''모델 로드 및 FT'''
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
model.cuda()

# Get all of the model's parameters as a list of tuples.
# 모델의 파라미터를 tuple 리스트 형태로 가져오기
params = list(model.named_parameters())  # @@ param을 가져오는 모델이 FC 부착되어야 함

# 계층별 param 수 출력
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

'''FT를 위한 환경 설정'''
# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # @@

# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

'''accuracy 반환하는 함수'''


# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):
    print(preds)  # @@
    print(np.argmax(preds, axis=1))  # @@
    print(np.argmax(preds, axis=1).flatten())  # @@
    pred_flat = np.argmax(preds, axis=1).flatten()
    print(pred_flat)
    labels_flat = labels.flatten()
    print(labels)  # @@
    print(labels_flat)  # @@
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


'''keep
for label in labels:
    def flat_accuracy(preds,label):
        pred=np.argmax(preds,axis=1)
        print(pred)
        return np.sum(pred==label)/len(labels)
'''

# pred_flat = np.argmax(preds, axis=1).flatten()
# labels_flat = labels.flatten()
# return np.sum(pred_flat == labels_flat) / len(labels_flat)
# import datetime
# def format_time(elapsed):
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss
#     '''
#     # Round to the nearest second.
#     elapsed_rounded = int(round((elapsed)))
#     # Format as hh:mm:ss
#     return str(datetime.timedelta(seconds=elapsed_rounded))
# print(pred_flat)
# print(labels_flat)

# we are ready to kick off the training


from babel.dates import format_date, format_datetime, format_time

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.

total_t0 = time.time()

num_labels = 8
hidden_size = 6768
dropout = torch.nn.Dropout()
classifier = torch.nn.Linear(hidden_size, 8)  # num_labels=8
loss_fct = torch.nn.CrossEntropyLoss()  # multi label classification을 위해 cross entropy loss를 사용
pooled_output = dataset

# For each epoch...
for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    num_labels = 8
    hidden_size = 6768
    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            #  elapsed = format_time(time.time() - t0)
            #  # Report progress.
            #  print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed)
            print('batch{:>5,} of {:>5,}.'.format(step, len(train_dataloader)))
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()
        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.

        print(len(pooled_output))  # 10
        pooled_output = dropout(torch.tensor(pooled_output))
        print(pooled_output)
        print(type(pooled_output))  # torch.Tensor이다.
        logits = classifier(pooled_output)
        print(logits.type)  # Tensor object
        print(type(labels))  # list
        print(logits.shape)  # [1,8]이다.
        print(labels)
        print(len(labels))  # 10이다.
        # labels.reshape(1,-1)
        # print(labels.shape)
        labels = torch.tensor(labels)
        print(labels.shape)
        print(logits.shape)  # torch.Size가 [10,1,8]
        loss = loss_fct(logits.view(-1, num_labels), torch.tensor(labels))
        print('logits:', logits)
        print('loss:', loss)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    # training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    # print("  Training epcoh took: {:}".format(training_time))
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    num_labels = 8
    hidden_size = 6768

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            # 여기 추가한 코드
            pooled_output = dropout(torch.tensor(pooled_output))
            logits = classifier(torch.tensor(pooled_output))
            print(logits.type)  # Tensor object
            print(logits.shape)  # [1,8]이다.
            print(labels)  # tensor(7)이다.
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            print('logits:', logits)
            print('loss:', loss)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        # total_eval_accuracy += flat_accuracy(logits, labels)

    # Report the final accuracy for this validation run.
    # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    try:
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    except ZeroDivisionError:
        print('ZeroDivision')

    # Calculate the average loss over all of the batches.
    # avg_val_loss = total_eval_loss / len(validation_dataloader)
    try:
        avg_val_loss = total_eval_loss / len(validation_dataloader)
    except ZeroDivisionError:
        print('ZeroDivision')
    # Measure how long the validation run took.
    # validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    # print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            # 'Training Time': training_time,
            # 'Validation Time': validation_time
        }
    )
print(training_stats[0]['epoch'])  # 1이다.
print(training_stats[0]['Training Loss'])
print(training_stats[0]['Valid. Loss'])

print("")
print("Training complete!")
# print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))