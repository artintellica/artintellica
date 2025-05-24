from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # A common BERT variant
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(["I love cats", "I adore kittens"], return_tensors="pt", padding=True)
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)  # Average embeddings
similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(f"Similarity: {similarity.item()}")
