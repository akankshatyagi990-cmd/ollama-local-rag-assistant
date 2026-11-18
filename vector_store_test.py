from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# VECTOR STORE
# -----------------------------

vector_store = InMemoryVectorStore(
    embedding=embeddings
)


# -----------------------------
# SAMPLE DOCUMENTS
# -----------------------------

documents = [
    "Random Forest uses many decision trees and combines their predictions.",

    "XGBoost builds trees sequentially, where each new tree tries to correct previous errors.",

    "Gradient Descent is an optimization algorithm used to minimize a loss function.",

    "Overfitting happens when a machine learning model performs well on training data but poorly on new data.",

    "Pizza is a popular food made with dough, sauce, cheese, and toppings."
]


# -----------------------------
# ADD TEXT TO VECTOR STORE
# -----------------------------

vector_store.add_texts(documents)


# -----------------------------
# USER QUESTION
# -----------------------------

question = "Which algorithm uses multiple decision trees?"


# -----------------------------
# SEMANTIC SEARCH
# -----------------------------

results = vector_store.similarity_search(
    question,
    k=2
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("Question:")
print(question)

print("\nMost Relevant Results:")

for index, document in enumerate(results, start=1):

    print(f"\nResult {index}:")
    print(document.page_content)