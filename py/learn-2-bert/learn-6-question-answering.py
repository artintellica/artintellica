from transformers import pipeline

# Load the pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Context and questions
context = "The Eiffel Tower is in Paris, France, and was completed in 1889."
questions = [
    "Where is the Eiffel Tower?",
    "When was the Eiffel Tower completed?"
]

# Run predictions
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (Confidence: {result['score']:.2f})\n")
