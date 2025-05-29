import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a dictionary of 3 "word" vectors (3D, random values between -1 and 1)
word_vectors = {
    "word1": np.random.uniform(low=-1, high=1, size=3),
    "word2": np.random.uniform(low=-1, high=1, size=3),
    "word3": np.random.uniform(low=-1, high=1, size=3),
}


# Function to compute cosine similarity
def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot_product / (norm_u * norm_v) if norm_u * norm_v != 0 else 0


# Initialize a 3x3 similarity matrix
words = list(word_vectors.keys())
sim_matrix = np.zeros((3, 3))

# Compute pairwise cosine similarities
for i, word_i in enumerate(words):
    for j, word_j in enumerate(words):
        sim_matrix[i, j] = cosine_similarity(word_vectors[word_i], word_vectors[word_j])

# Print word vectors
print("Word vectors:")
for word, vector in word_vectors.items():
    print(f"{word}: {vector}")

# Print similarity matrix
print("\nCosine similarity matrix:")
print(np.array2string(sim_matrix, precision=4, suppress_small=True))
