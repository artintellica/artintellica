from transformers import pipeline

# Load the pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Text and candidate labels
text = "This movie was thrilling and kept me on edge!"
labels = ["happy", "sad", "neutral"]

# Run predictions
result = classifier(text, labels)
print(f"Text: {text}")
for label, score in zip(result["labels"], result["scores"]):
    print(f"Label: {label} (Confidence: {score:.2f})")
