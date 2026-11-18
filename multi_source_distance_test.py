from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# -----------------------------
# SETTINGS
# -----------------------------

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "local_rag"

EMBEDDING_MODEL = "nomic-embed-text"

QUESTION = "What does Gradient Descent do?"


# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# -----------------------------
# LOAD CHROMA DATABASE
# -----------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)


# -----------------------------
# SEARCH ALL CHUNKS
# -----------------------------

results = vector_store.similarity_search_with_score(
    QUESTION,
    k=12
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("Question:")
print(QUESTION)

print("\nAll Retrieval Results:")


for index, (document, distance) in enumerate(
    results,
    start=1
):

    print(
        f"\n========== Result {index} =========="
    )

    print(
        "Distance:",
        round(distance, 4)
    )

    print(
        "File:",
        document.metadata.get(
            "source",
            "Unknown"
        )
    )

    print(
        "Type:",
        document.metadata.get(
            "file_type",
            "Unknown"
        )
    )

    if document.metadata.get(
        "file_type"
    ) == "pdf":

        print(
            "Page:",
            document.metadata.get(
                "page",
                "N/A"
            )
        )

    print(
        "Chunk:",
        document.metadata.get(
            "chunk_number",
            "N/A"
        )
    )

    print("\nContent:")
    print(
        document.page_content
    )