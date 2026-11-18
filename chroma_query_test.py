from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# SETTINGS
# -----------------------------

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "local_rag"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

MAX_DISTANCE = 0.90


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
# CHAT HISTORY
# -----------------------------

chat_history = []


# -----------------------------
# QUESTION REWRITING PROMPT
# -----------------------------

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Rewrite the user's current question into a standalone question.

        Use the conversation history only to understand references
        such as "it", "that", "this", or "they".

        Do not answer the question.

        Return only the rewritten question.
        """
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    (
        "human",
        "{question}"
    )
])


output_parser = StrOutputParser()


rewrite_chain = (
    rewrite_prompt
    | llm
    | output_parser
)


# -----------------------------
# RAG PROMPT
# -----------------------------

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a beginner-friendly AI Learning Assistant.

        Answer using only the provided document context.

        Do not use outside knowledge.

        If the answer is not available in the context, say:

        "I do not have enough information in the provided documents."

        Keep answers simple, concise and accurate.
        """
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    (
        "human",
        """
        Document Context:

        {context}

        Question:

        {question}
        """
    )
])


rag_chain = (
    rag_prompt
    | llm
    | output_parser
)


# -----------------------------
# CREATE RETRIEVAL QUERY
# -----------------------------

def create_retrieval_query(question):

    if not chat_history:
        return question

    rewritten_question = rewrite_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    return rewritten_question.strip()


# -----------------------------
# ANSWER QUESTION
# -----------------------------

def answer_question(question):

    retrieval_query = create_retrieval_query(
        question
    )

    results = vector_store.similarity_search_with_score(
        retrieval_query,
        k=3
    )

    relevant_results = [
        (document, distance)
        for document, distance in results
        if distance <= MAX_DISTANCE
    ]

    if not relevant_results:

        return (
            "I do not have enough information "
            "in the provided documents.",
            results[0][1],
            [],
            retrieval_query
        )

    context = "\n\n".join(
        document.page_content
        for document, distance in relevant_results
    )

    response = rag_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    return (
        response,
        results[0][1],
        relevant_results,
        retrieval_query
    )


# -----------------------------
# START CHATBOT
# -----------------------------

print("Conversational Chroma RAG Assistant")

print(
    "Database:",
    PERSIST_DIRECTORY
)

print(
    "Maximum Distance:",
    MAX_DISTANCE
)

print("\nCommands:")
print("exit / quit")
print("clear chat")

print()


while True:

    question = input("You: ")

    if question.lower() in [
        "exit",
        "quit"
    ]:

        print("Bot: Goodbye!")
        break

    if question.lower() == "clear chat":

        chat_history.clear()

        print(
            "\nBot: Conversation memory cleared.\n"
        )

        continue

    response, best_distance, sources, retrieval_query = answer_question(
        question
    )

    # -----------------------------
    # SAVE CHAT HISTORY
    # -----------------------------

    chat_history.append(
        HumanMessage(
            content=question
        )
    )

    chat_history.append(
        AIMessage(
            content=response
        )
    )

    # -----------------------------
    # DISPLAY ANSWER
    # -----------------------------

    print("\nRetrieval Query:")
    print(retrieval_query)

    print("\nBot:")
    print(response)

    print(
        "\nBest Distance:",
        round(best_distance, 4)
    )

    if sources:

        print("\nSources:")

        for index, (
            document,
            distance
        ) in enumerate(
            sources,
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
                "Chunk:",
                document.metadata["chunk_number"]
            )

            print(
                "Distance:",
                round(distance, 4)
            )

    print()