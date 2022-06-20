#!pip install -q git+https://github.com/huggingface/transformers.git
import math
import torch
from transformers.activations import gelu
from transformers import (BertTokenizer, BertConfig,
                          BertForSequenceClassification, BertPreTrainedModel,
                          apply_chunking_to_forward, set_seed,
                          )
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,
                                           BaseModelOutputWithPoolingAndCrossAttentions,
                                           SequenceClassifierOutput,
                                           )

# Set seed for reproducibility.
set_seed(123)

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = 2

# GELU Activation function.
ACT2FN = {"gelu": gelu}

# Define BertLayerNorm.
BertLayerNorm = torch.nn.LayerNorm

input_texts = ['I love cats!',
              "He hates pineapple pizza."]

# Sentiment labels
labels = [1,0]

# Create BertTokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Create input sequence using tokenizer.
input_sequences = tokenizer(text=input_texts, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
print('label붙이기전:',input_sequences)
# Since input_sequence is a dictionary we can also add the labels to it
# want to make sure all values ar tensors.
input_sequences.update({'labels':torch.tensor(labels)})

print('label붙인 후:',input_sequences) #이러면

# The tokenizer will return a dictionary of three: input_ids, attention_mask and token_type_ids.
# Let's do a pretty print.
print('PRETTY PRINT OF `input_sequences` UPDATED WITH `labels`:')
[print('%s : %s\n'%(k,v)) for k,v in input_sequences.items()];

# Lets see how the text looks like after Bert Tokenizer.
# We see the special tokens added.
print('ORIGINAL TEXT:')
[print(example) for example in input_texts];
print('\nTEXT AFTER USING `BertTokenizer`:')
[print(tokenizer.decode(example)) for example in input_sequences['input_ids'].numpy()];
bert_configuration = BertConfig.from_pretrained('bert-base-cased')

# create Bert model with classification layer - BertForSequenceClassificatin
bert_for_sequence_classification_model = BertForSequenceClassification(bert_configuration)