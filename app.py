import json
import os

import streamlit as st

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "local_rag"

CHAT_HISTORY_FILE = "rag_chat_history.json"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

TXT_MAX_DISTANCE = 0.75
PDF_MAX_DISTANCE = 0.65

DISTANCE_MARGIN = 0.18


# --------------------------------------------------
# LOAD RAG COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_rag_system():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.1
    )

    return vector_store, llm


vector_store, llm = load_rag_system()


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a beginner-friendly AI Learning Assistant.

        Answer using only the provided document context.

        Do not use outside knowledge.

        Do not combine unrelated information.

        If the answer is not explicitly supported by the context, say:

        "I do not have enough information in the provided documents."

        Keep the answer simple, concise and accurate.
        """
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


output_parser = StrOutputParser()


rag_chain = (
    prompt
    | llm
    | output_parser
)


# --------------------------------------------------
# LOAD CHAT HISTORY
# --------------------------------------------------

def load_chat_history():

    if not os.path.exists(
        CHAT_HISTORY_FILE
    ):

        return []

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

    except (
        json.JSONDecodeError,
        OSError
    ):

        pass

    return []


# --------------------------------------------------
# SAVE CHAT HISTORY
# --------------------------------------------------

def save_chat_history():

    with open(
        CHAT_HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            st.session_state.messages,
            file,
            indent=4,
            ensure_ascii=False
        )


# --------------------------------------------------
# SESSION MEMORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = (
        load_chat_history()
    )


# --------------------------------------------------
# FOLLOW-UP DETECTION
# --------------------------------------------------

def is_follow_up_question(question):

    question_lower = (
        question.lower().strip()
    )

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
        for starter
        in follow_up_starters
    )


# --------------------------------------------------
# EXTRACT MAIN TOPIC
# --------------------------------------------------

def extract_main_topic(question):

    parts = [
        part.strip()
        for part in question.split("?")
        if part.strip()
    ]

    if parts:

        return parts[0] + "?"

    return question


# --------------------------------------------------
# GET LAST STANDALONE TOPIC
# --------------------------------------------------

def get_last_standalone_topic():

    for message in reversed(
        st.session_state.messages
    ):

        if (
            message.get("role")
            == "user"
        ):

            previous_question = (
                message.get(
                    "content",
                    ""
                )
            )

            if not is_follow_up_question(
                previous_question
            ):

                return extract_main_topic(
                    previous_question
                )

    return None


# --------------------------------------------------
# CREATE RETRIEVAL QUERY
# --------------------------------------------------

def create_retrieval_query(question):

    if not is_follow_up_question(
        question
    ):

        return question

    main_topic = (
        get_last_standalone_topic()
    )

    if main_topic:

        return (
            main_topic
            + " "
            + question
        )

    return question


# --------------------------------------------------
# CHECK DISTANCE
# --------------------------------------------------

def passes_absolute_threshold(
    document,
    distance
):

    file_type = (
        document.metadata.get(
            "file_type",
            "txt"
        )
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


# --------------------------------------------------
# RAG FUNCTION
# --------------------------------------------------

def answer_question(question):

    retrieval_query = (
        create_retrieval_query(
            question
        )
    )

    results = (
        vector_store
        .similarity_search_with_score(
            retrieval_query,
            k=6
        )
    )

    if not results:

        return (
            "I do not have enough information "
            "in the provided documents.",
            [],
            retrieval_query
        )

    best_distance = results[0][1]

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

        if (
            absolute_ok
            and relative_ok
        ):

            relevant_results.append(
                (
                    document,
                    distance
                )
            )

    if not relevant_results:

        return (
            "I do not have enough information "
            "in the provided documents.",
            [],
            retrieval_query
        )

    context = "\n\n".join(
        document.page_content
        for document, distance
        in relevant_results
    )

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    return (
        response,
        relevant_results,
        retrieval_query
    )


# --------------------------------------------------
# DISPLAY SOURCES
# --------------------------------------------------

def display_sources(source_list):

    for index, source_data in enumerate(
        source_list,
        start=1
    ):

        with st.expander(
            f"Source {index}: "
            f"{source_data['source']}"
        ):

            st.write(
                f"**File Type:** "
                f"{source_data['file_type']}"
            )

            if (
                source_data["file_type"]
                == "pdf"
            ):

                st.write(
                    f"**Page:** "
                    f"{source_data['page']}"
                )

            st.write(
                f"**Chunk:** "
                f"{source_data['chunk']}"
            )

            st.write(
                f"**Distance:** "
                f"{source_data['distance']:.4f}"
            )

            st.write(
                "**Retrieved Content:**"
            )

            st.write(
                source_data["content"]
            )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "🤖 Local RAG Assistant"
)

st.caption(
    "Ask questions from your local "
    "TXT and PDF knowledge base."
)

st.info(
    "Powered locally by Ollama, "
    "LangChain and Chroma."
)


# --------------------------------------------------
# CLEAR CHAT
# --------------------------------------------------

if st.button("Clear Chat"):

    st.session_state.messages = []

    if os.path.exists(
        CHAT_HISTORY_FILE
    ):

        os.remove(
            CHAT_HISTORY_FILE
        )

    st.rerun()


# --------------------------------------------------
# DISPLAY SAVED MESSAGES
# --------------------------------------------------

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message.get(
            "role",
            "assistant"
        )
    ):

        st.markdown(
            message.get(
                "content",
                ""
            )
        )

        if message.get(
            "retrieval_query"
        ):

            st.caption(
                "Retrieval Query: "
                + message[
                    "retrieval_query"
                ]
            )

        if message.get(
            "sources"
        ):

            display_sources(
                message["sources"]
            )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question from your documents..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.messages.append(
        user_message
    )

    save_chat_history()


    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching knowledge base..."
        ):

            (
                response,
                sources,
                retrieval_query
            ) = answer_question(
                question
            )


        st.markdown(response)

        st.caption(
            "Retrieval Query: "
            + retrieval_query
        )


        source_list = []


        for document, distance in sources:

            source_data = {
                "source": (
                    document.metadata.get(
                        "source",
                        "Unknown"
                    )
                ),

                "file_type": (
                    document.metadata.get(
                        "file_type",
                        "Unknown"
                    )
                ),

                "page": (
                    document.metadata.get(
                        "page",
                        "N/A"
                    )
                ),

                "chunk": (
                    document.metadata.get(
                        "chunk_number",
                        "N/A"
                    )
                ),

                "distance": float(
                    distance
                ),

                "content": (
                    document.page_content
                )
            }

            source_list.append(
                source_data
            )


        if source_list:

            display_sources(
                source_list
            )


    assistant_message = {
        "role": "assistant",
        "content": response,
        "sources": source_list,
        "retrieval_query": retrieval_query
    }

    st.session_state.messages.append(
        assistant_message
    )

    save_chat_history()