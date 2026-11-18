import json
import os

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

CHAT_HISTORY_FILE = "rag_chat_history.json"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

TXT_MAX_DISTANCE = 0.75
PDF_MAX_DISTANCE = 0.65

DISTANCE_MARGIN = 0.18


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
# LOAD CHAT HISTORY
# -----------------------------

def load_chat_history():

    history = []

    if not os.path.exists(
        CHAT_HISTORY_FILE
    ):
        return history

    with open(
        CHAT_HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        saved_messages = json.load(
            file
        )

    for message in saved_messages:

        if message["role"] == "user":

            history.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            history.append(
                AIMessage(
                    content=message["content"]
                )
            )

    return history


# -----------------------------
# SAVE CHAT HISTORY
# -----------------------------

def save_chat_history():

    saved_messages = []

    for message in chat_history:

        if isinstance(
            message,
            HumanMessage
        ):

            saved_messages.append({
                "role": "user",
                "content": message.content
            })

        elif isinstance(
            message,
            AIMessage
        ):

            saved_messages.append({
                "role": "assistant",
                "content": message.content
            })

    with open(
        CHAT_HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            saved_messages,
            file,
            indent=4,
            ensure_ascii=False
        )


chat_history = load_chat_history()


# -----------------------------
# RAG PROMPT
# -----------------------------

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a beginner-friendly AI Learning Assistant.

        Answer using only the provided document context.

        Use conversation history only to understand references
        in follow-up questions.

        Do not use outside knowledge.

        Do not combine unrelated information from different topics.

        If the answer is not explicitly supported by the context, say:

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

        Current Question:

        {question}
        """
    )
])


output_parser = StrOutputParser()


rag_chain = (
    rag_prompt
    | llm
    | output_parser
)


# -----------------------------
# FOLLOW-UP DETECTION
# -----------------------------

def is_follow_up_question(question):

    question_lower = question.lower().strip()

    follow_up_starters = [
        "is it ",
        "does it ",
        "can it ",
        "could it ",
        "would it ",
        "how does it ",
        "why does it ",
        "when does it ",
        "where does it ",
        "what about it",
        "is that ",
        "does that ",
        "can that ",
        "what about that",
        "are they ",
        "do they ",
        "can they ",
        "how do they ",
        "what about them"
    ]

    return any(
        question_lower.startswith(
            starter
        )
        for starter in follow_up_starters
    )


# -----------------------------
# EXTRACT MAIN TOPIC
# -----------------------------

def extract_main_topic(question):

    parts = [
        part.strip()
        for part in question.split("?")
        if part.strip()
    ]

    if parts:
        return parts[0] + "?"

    return question


# -----------------------------
# GET LAST STANDALONE TOPIC
# -----------------------------

def get_last_standalone_topic():

    for message in reversed(
        chat_history
    ):

        if isinstance(
            message,
            HumanMessage
        ):

            previous_question = message.content

            if not is_follow_up_question(
                previous_question
            ):

                return extract_main_topic(
                    previous_question
                )

    return None


# -----------------------------
# CREATE RETRIEVAL QUERY
# -----------------------------

def create_retrieval_query(question):

    if not is_follow_up_question(
        question
    ):
        return question


    main_topic = get_last_standalone_topic()


    if main_topic:

        return (
            main_topic
            + " "
            + question
        )


    return question


# -----------------------------
# CHECK ABSOLUTE DISTANCE
# -----------------------------

def passes_absolute_threshold(
    document,
    distance
):

    file_type = document.metadata.get(
        "file_type",
        "txt"
    )


    if file_type == "pdf":

        return (
            distance
            <= PDF_MAX_DISTANCE
        )


    return (
        distance
        <= TXT_MAX_DISTANCE
    )


# -----------------------------
# ANSWER QUESTION
# -----------------------------

def answer_question(question):

    retrieval_query = create_retrieval_query(
        question
    )


    results = vector_store.similarity_search_with_score(
        retrieval_query,
        k=6
    )


    if not results:

        return (
            "I do not have enough information "
            "in the provided documents.",
            [],
            retrieval_query
        )


    # Best result has the smallest distance
    best_distance = results[0][1]


    # Only accept results reasonably close
    # to the best result
    relative_limit = (
        best_distance
        + DISTANCE_MARGIN
    )


    relevant_results = []


    for document, distance in results:

        absolute_ok = (
            passes_absolute_threshold(
                document,
                distance
            )
        )

        relative_ok = (
            distance
            <= relative_limit
        )


        if absolute_ok and relative_ok:

            relevant_results.append(
                (
                    document,
                    distance
                )
            )


    # -----------------------------
    # NO GOOD CONTEXT
    # -----------------------------

    if not relevant_results:

        return (
            "I do not have enough information "
            "in the provided documents.",
            [],
            retrieval_query
        )


    # -----------------------------
    # BUILD CONTEXT
    # -----------------------------

    context = "\n\n".join(
        document.page_content
        for document, distance
        in relevant_results
    )


    # -----------------------------
    # GENERATE ANSWER
    # -----------------------------

    response = rag_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })


    return (
        response,
        relevant_results,
        retrieval_query
    )


# -----------------------------
# START CHATBOT
# -----------------------------

print(
    "Local Multi-Document RAG Assistant"
)

print(
    "TXT Maximum Distance:",
    TXT_MAX_DISTANCE
)

print(
    "PDF Maximum Distance:",
    PDF_MAX_DISTANCE
)

print(
    "Distance Margin:",
    DISTANCE_MARGIN
)

print(
    "Saved Messages:",
    len(chat_history)
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

        if os.path.exists(
            CHAT_HISTORY_FILE
        ):

            os.remove(
                CHAT_HISTORY_FILE
            )

        print(
            "\nBot: Conversation memory cleared.\n"
        )

        continue


    response, sources, retrieval_query = answer_question(
        question
    )


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

    save_chat_history()


    print("\nRetrieval Query:")
    print(retrieval_query)


    print("\nBot:")
    print(response)


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


            print(
                "Distance:",
                round(
                    distance,
                    4
                )
            )


    print()