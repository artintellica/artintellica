from transformers import pipeline
from PIL import Image

# Load the pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Load an image (replace 'cat.jpg' with a local image file)
image = Image.open("cat.jpg")

# Run predictions
results = classifier(image)
for result in results[:3]:  # Top 3 predictions
    print(f"Label: {result['label']} (Confidence: {result['score']:.2f})")
