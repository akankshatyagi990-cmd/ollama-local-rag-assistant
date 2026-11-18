from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# -----------------------------
# SETTINGS
# -----------------------------

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "local_rag"

PDF_FILE = "sample_knowledge.pdf"

EMBEDDING_MODEL = "nomic-embed-text"


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
# TEST QUESTIONS
# -----------------------------

questions = [
    "What is Random Forest?",
    "What is Machine Learning?"
]


# -----------------------------
# SEARCH PDF
# -----------------------------

for question in questions:

    print("\n======================================")
    print("Question:")
    print(question)
    print("======================================")

    results = vector_store.similarity_search_with_score(
        question,
        k=3,
        filter={
            "source": PDF_FILE
        }
    )


    for index, (
        document,
        distance
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {index}"
        )

        print(
            "Distance:",
            round(distance, 4)
        )

        print(
            "Page:",
            document.metadata.get(
                "page",
                "N/A"
            )
        )

        print(
            "Chunk:",
            document.metadata[
                "chunk_number"
            ]
        )

        print("Content:")
        print(
            document.page_content
        )