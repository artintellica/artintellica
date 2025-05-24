from transformers import pipeline

# Load the pipeline
generator = pipeline("text-generation", model="distilgpt2")

# Generate text
prompt = "Once upon a time in a magical forest,"
result = generator(prompt, max_length=50, num_return_sequences=3)

# Print results
for i, text in enumerate(result):
    print(f"Generated Text {i+1}: {text['generated_text']}\n")
