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
#first,we need to define dataloaders, which we will use to iterate over batches.
#we just need to apply a bit post-processing to our tokenized_datasets before doing that to:
#-remove the columns corresponding to values the model does not expect (여기서는 text 열이다.)
#-rename column 'label' to 'labels' (왜냐하면 model expect the argument to be named 'labels')
#-set the format of datasets so they return Pytorch Tensors instead of lists (리스트 대신 tensor를 반환해야함.)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset=tokenized_dataset['train'].shuffle(seed=42).select(range(1000))
small_eval_dataset=tokenized_dataset['test'].shuffle(seed=42).select(range(1000))

#Now that this is done, we can easily define our dataloaders:
from torch.utils.data import DataLoader
train_dataloader=DataLoader(small_train_dataset,shuffle=True,batch_size=8)
eval_dataloader=DataLoader(small_eval_dataset,batch_size=8)
#Next, we define our model:
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

#we are almost ready to write our training loop인데, 지금 빠진 것이 optimizer와 learning rate scheduler이다.
#기본 optimizer: AdamW이다.
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

#finally, the learning rate scheduler used by default is just a linear decay from maximum value(5e-5) to 0:
from transformers import get_scheduler
num_epochs=3
num_training_stesps=num_epochs*len(train_dataloader)
lr_scheduler=get_scheduler("linear",optimizer=optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)

#One last thing, we will want to use GPU if we have access to one. To do this, we define a device we will put our model and our batches on.
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#We now are ready to train.
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
	for batch in train_dataloader:
		batch={k:v.to(device) for k,v in batch.items()}
		outputs=model(**batch)
		loss=outputs.loss
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		optimizer.zero_grad()
		progress_bar.update(1)

#이제 결과를 체크하기 위해서 evaluation loop를 적어보자. trainer section처럼 datasets library로부터 metric을 사용할 것이다.
metric= load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()










