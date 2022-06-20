#from transformers import BertModel
from tokenization_kobert import KoBertTokenizer

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
model = BertModel.from_pretrained("monologg/kobert")
sentence='나는 학교에 간다.'
tokens=tokenizer.tokenize(sentence)
print(tokens)

#inputs=tokenizer('보험 하나만 들어줄래?',return_tensors='pt')

