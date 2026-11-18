from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# SETTINGS
# -----------------------------

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "local_rag"

PDF_FILE = "sample_knowledge.pdf"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

MAX_PDF_DISTANCE = 0.60


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
# LOCAL LLM
# -----------------------------

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.1
)


# -----------------------------
# RAG PROMPT
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a beginner-friendly AI assistant.

        Answer using only the provided PDF context.

        Do not use outside knowledge.

        If the answer is not explicitly supported by the context, say:

        "I do not have enough information in the PDF."

        Keep the answer simple, concise and accurate.
        """
    ),
    (
        "human",
        """
        PDF Context:

        {context}

        Question:

        {question}
        """
    )
])


output_parser = StrOutputParser()

chain = (
    prompt
    | llm
    | output_parser
)


# -----------------------------
# START PDF RAG
# -----------------------------

print("PDF RAG Assistant")
print("PDF:", PDF_FILE)
print("Maximum Distance:", MAX_PDF_DISTANCE)

print("\nType 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() in [
        "exit",
        "quit"
    ]:
        print("Bot: Goodbye!")
        break


    # -----------------------------
    # SEARCH PDF
    # -----------------------------

    results = vector_store.similarity_search_with_score(
        question,
        k=3,
        filter={
            "source": PDF_FILE
        }
    )


    if not results:

        print(
            "\nBot:\n"
            "I do not have enough information in the PDF.\n"
        )

        continue


    # -----------------------------
    # FILTER BY DISTANCE
    # -----------------------------

    relevant_results = [
        (document, distance)
        for document, distance in results
        if distance <= MAX_PDF_DISTANCE
    ]


    # -----------------------------
    # NO GOOD MATCH
    # -----------------------------

    if not relevant_results:

        print(
            "\nBot:\n"
            "I do not have enough information in the PDF."
        )

        print(
            "\nBest Distance:",
            round(results[0][1], 4)
        )

        print()

        continue


    # -----------------------------
    # BUILD CONTEXT
    # -----------------------------

    context = "\n\n".join(
        document.page_content
        for document, distance in relevant_results
    )


    # -----------------------------
    # GENERATE ANSWER
    # -----------------------------

    response = chain.invoke({
        "context": context,
        "question": question
    })


    # -----------------------------
    # DISPLAY ANSWER
    # -----------------------------

    print("\nBot:")
    print(response)

    print(
        "\nBest Distance:",
        round(relevant_results[0][1], 4)
    )

    print("\nPDF Sources:")


    for index, (
        document,
        distance
    ) in enumerate(
        relevant_results,
        start=1
    ):

        print(
            f"\nSource {index}:"
        )

        print(
            "File:",
            document.metadata["source"]
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
            document.metadata["chunk_number"]
        )

        print(
            "Distance:",
            round(distance, 4)
        )

    print()