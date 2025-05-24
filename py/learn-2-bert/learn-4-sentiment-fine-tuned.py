from transformers import pipeline

# Load a fine-tuned sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test sentences
texts = [
    "I love this movie, it's amazing!",
    "This film was terrible and boring.",
    "It's okay, not great but not bad either.",
    "I had eggs for breakfast.",
]

# Run sentiment analysis
for text in texts:
    result = sentiment_pipeline(text)
    label = result[0]["label"]
    score = result[0]["score"]
    print(f"Text: {text}")
    print(f"Sentiment: {label} (Confidence: {score:.2f})\n")
