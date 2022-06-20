#classifier구현할 파일(11.30)
from transformers import BertTokenizer, BertModel
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
for i in tqdm(range(0,51)): #인덱스가 0부터 64977이다.
    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt')
    outputs=model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states)
    i+=1
#inputs=tokenizer(aihub_text,return_tensors='pt')
#inputs = tokenizer('나는 학교에 간다', return_tensors="pt")
#outputs = model(**inputs)
#print(list_last_hidden_states) #이러면 last_hidden_states를 쭉 저장한 파일이 되고,
#print(list_last_hidden_states[0][0]) #이게 last_hidden_states의 1번째
#print(list_last_hidden_states[0][0][0])

# Perform a forward pass. We only care about the output and no gradients.
# with torch.no_grad():
#   output = model.forward(**input_sequences)
#
# print()
#
# # Let's check how a forward pass output looks like.
# print('FORWARD PASS OUTPUT:', output)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



# create Bert model with classification layer - BertForSequenceClassification
bert_for_sequence_classification_model = BertForSequenceClassification(bert_configuration)

# perform forward pass on entire model
outputs = bert_for_sequence_classification_model(**input_sequences)

