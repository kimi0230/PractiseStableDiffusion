from transformers import pipeline

classifier = pipeline("zero-shot-classification")


res = classifier("This is a course about Python list comprehension",candidate_labels=["education", "politics","business"])

# {'sequence': 'This is a course about Python list comprehension', 'labels': ['education', 'business', 'politics'], 'scores': [0.9622027277946472, 0.02684134989976883, 0.010955979116261005]}
print(res) 