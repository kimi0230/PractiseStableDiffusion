from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# UserWarning: `return_all_scores` is now deprecated, if want a similar funcionality use `top_k=None` instead of `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`
res = classifier("I've been waiting for a HuggingFace course my whole life.", top_k=None)
print(res)