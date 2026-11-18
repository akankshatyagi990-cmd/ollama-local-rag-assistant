import math

from langchain_ollama import OllamaEmbeddings


# Create local embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Sentences to compare
sentence_1 = "Machine learning learns patterns from data."

sentence_2 = "AI models learn from examples."

sentence_3 = "I like eating pizza."


# Convert sentences into vectors
vector_1 = embeddings.embed_query(sentence_1)
vector_2 = embeddings.embed_query(sentence_2)
vector_3 = embeddings.embed_query(sentence_3)


# Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (
        magnitude_a * magnitude_b
    )


# Compare similarities
similarity_1_2 = cosine_similarity(
    vector_1,
    vector_2
)

similarity_1_3 = cosine_similarity(
    vector_1,
    vector_3
)


print("Sentence 1:")
print(sentence_1)

print("\nSentence 2:")
print(sentence_2)

print("\nSentence 3:")
print(sentence_3)


print("\nSimilarity:")

print(
    "Sentence 1 vs Sentence 2:",
    round(similarity_1_2, 4)
)

print(
    "Sentence 1 vs Sentence 3:",
    round(similarity_1_3, 4)
)