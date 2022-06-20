model_bert = BertModel.from_pretrained('bert-base-multilingual-uncased') #고쳐야함.
print(model_bert)

from pytorch_transformers import BertTokenizer, BertForSequenceClassification
print(BertForSequenceClassification)
BertForSequenceClassification

model_bertforclassification = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 8, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
#model.cuda()
print(model_bertforclassification)

