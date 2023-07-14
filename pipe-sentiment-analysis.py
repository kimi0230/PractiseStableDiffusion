from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

# UserWarning: `return_all_scores` is now deprecated, if want a similar funcionality use `top_k=None` instead of `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`
res = classifier("I've been waiting for a HuggingFace course my whole life.", top_k=None)

# [{'label': 'POSITIVE', 'score': 0.9598051905632019}, {'label': 'NEGATIVE', 'score': 0.04019483923912048}]
print(res) 

# Specifying a custom model and tokenizer for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# Loading the tokenizer associated with the model
mode = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=mode, tokenizer=tokenizer)
res = classifier("I've been waiting for a HuggingFace course my whole life.", top_k=None)

# [{'label': 'POSITIVE', 'score': 0.9598051905632019}, {'label': 'NEGATIVE', 'score': 0.04019483923912048}]
print(res) 

sequence = "Using a Transformer network is simple"

res =tokenizer(sequence)
# {'input_ids': [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
# 101 is the token id for [CLS] and 102 is the token id for [SEP]
# "[CLS]" 標記是在輸入序列的開始處插入的特殊標記，代表「分類」（classification）。它用於指示模型開始進行任務預測或序列分類的地方。在訓練期間，模型會學習從 "[CLS]"標記中獲取序列級別的信息。
# "[SEP]" 標記是用於分隔不同句子或文本片段的特殊標記，代表「分隔」（separator）。它通常用於需要處理多個句子的任務，例如句子對分類、問答系統等。"[SEP]" 標記的存在有助於模型理解多個句子之間的邏輯關係。
print(res)

tokens = tokenizer.tokenize(sequence)
# ['using', 'a', 'transform', '##er', 'network', 'is', 'simple']
print(tokens)

ids=tokenizer.convert_tokens_to_ids(tokens)
# [2478, 1037, 10938, 2121, 2897, 2003, 3722]
print(ids)

decoded_striing = tokenizer.decode(ids)
# using a transformer network is simple
print(decoded_striing)