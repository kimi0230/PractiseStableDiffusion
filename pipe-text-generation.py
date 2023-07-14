from transformers import pipeline

classifier = pipeline("text-generation", model="distilgpt2")


res = classifier("In this course, we will teach you how to",max_length=50, do_sample=True, top_k=50, top_p=0.95, temperature=0.7, no_repeat_ngram_size=2, num_return_sequences=1, early_stopping=True, use_cache=True, num_beams=5, prefix="In this course, we will teach you how to")

# [{'generated_text': 'In this course, we will teach you how to Inject yourself in the same way you do in your life.'}]
print(res) 