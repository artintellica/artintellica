from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # A common BERT variant
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

texts = ["I love this movie!", "This movie is great!", "Awesome!", "This is awful.", "It's okay, I guess."]
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    print(f"Text: {text} => Sentiment: {'Positive' if predicted_class == 1 else 'Negative'}")
