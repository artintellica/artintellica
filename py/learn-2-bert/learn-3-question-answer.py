from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "The capital of France is Paris."
question = "What is the capital of France?"
result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}")
