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