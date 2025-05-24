from transformers import pipeline

# Load the pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Test texts
texts = [
    "This movie is absolutely fantastic!",
    "It was okay, nothing special.",
    "I hated this film, total waste of time.",
]

# Run predictions
for text in texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2f})\n")
