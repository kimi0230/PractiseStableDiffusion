from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Specifying a custom model and tokenizer for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# Loading the tokenizer associated with the model
mode = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=mode, tokenizer=tokenizer)

# List of input texts for sentiment analysis
X_train = ["I've been waiting for a HuggingFace course my whole life.","Golang is great"]

# Performing sentiment analysis on the input texts using the pipeline
res = classifier(X_train)
# [{'label': 'POSITIVE', 'score': 0.9598051905632019}, {'label': 'POSITIVE', 'score': 0.9998632669448853}]
print(res) 

# Tokenizing the input texts for the model
batch = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")

# {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
#           2607,  2026,  2878,  2166,  1012,   102],
#         [  101,  2175, 25023,  2003,  2307,   102,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
print(batch)

with torch.no_grad():
    # Passing the tokenized input texts as input to the model
    outputs = mode(**batch)
    # SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123], [-4.2678,  4.6292]]), hidden_states=None, attentions=None)
    print(outputs)
    
    # Applying softmax function to the logits to get predicted probabilities
    predictions = F.softmax(outputs.logits, dim=-1)
    # Output: tensor([[4.0195e-02, 9.5981e-01], [1.3677e-04, 9.9986e-01]])
    print(predictions)
    
    # Getting the predicted labels by selecting the label with maximum probability
    labels = torch.argmax(predictions, dim=-1)
    # Output: tensor([1, 1])
    print("labels", labels)