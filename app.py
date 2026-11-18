import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

UPLOAD_FOLDER = Path(
    "knowledge/uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

TXT_MAX_DISTANCE = 0.75
PDF_MAX_DISTANCE = 0.65

DISTANCE_MARGIN = 0.18

BATCH_SIZE = 20


# --------------------------------------------------
# TEXT SPLITTER
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


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

        if isinstance(
            data,
            list
        ):

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

def is_follow_up_question(
    question
):

    question_lower = (
        question
        .lower()
        .strip()
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

def extract_main_topic(
    question
):

    parts = [
        part.strip()
        for part
        in question.split("?")
        if part.strip()
    ]

    if parts:

        return (
            parts[0]
            + "?"
        )

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

def create_retrieval_query(
    question
):

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
# RAG QUESTION FUNCTION
# --------------------------------------------------

def answer_question(
    question
):

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

    best_distance = (
        results[0][1]
    )

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
# CREATE FILE HASH
# --------------------------------------------------

def create_file_hash(
    file_bytes
):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# --------------------------------------------------
# DELETE EXISTING SOURCE
# --------------------------------------------------

def delete_existing_source(
    source_name
):

    existing = vector_store.get(
        where={
            "source": source_name
        }
    )

    existing_ids = (
        existing.get(
            "ids",
            []
        )
    )

    if existing_ids:

        vector_store.delete(
            ids=existing_ids
        )

        return len(
            existing_ids
        )

    return 0


# --------------------------------------------------
# DELETE UPLOADED DOCUMENT
# --------------------------------------------------

def delete_uploaded_document(
    source_name
):

    existing = vector_store.get(
        where={
            "source": source_name
        },
        include=[
            "metadatas"
        ]
    )

    existing_ids = (
        existing.get(
            "ids",
            []
        )
    )

    metadatas = (
        existing.get(
            "metadatas",
            []
        )
    )


    if not existing_ids:

        return {
            "status": "not_found",
            "deleted_chunks": 0
        }


    uploaded_flags = [
        metadata.get(
            "uploaded",
            False
        )
        for metadata in metadatas
        if metadata is not None
    ]


    if not uploaded_flags:

        return {
            "status": "protected",
            "deleted_chunks": 0
        }


    if not all(
        uploaded_flags
    ):

        return {
            "status": "protected",
            "deleted_chunks": 0
        }


    vector_store.delete(
        ids=existing_ids
    )


    file_path = (
        UPLOAD_FOLDER
        / source_name
    )


    if file_path.exists():

        try:

            file_path.unlink()

        except OSError:

            pass


    return {
        "status": "deleted",
        "deleted_chunks": len(
            existing_ids
        )
    }


# --------------------------------------------------
# CHECK DUPLICATE FILE
# --------------------------------------------------

def file_already_indexed(
    file_hash
):

    existing = vector_store.get(
        where={
            "file_hash": file_hash
        }
    )

    existing_ids = (
        existing.get(
            "ids",
            []
        )
    )

    return bool(
        existing_ids
    )


# --------------------------------------------------
# TXT CHUNKS
# --------------------------------------------------

def extract_txt_chunks(
    file_path,
    file_hash
):

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    chunks = (
        text_splitter.split_text(
            text
        )
    )

    texts = []
    metadatas = []
    ids = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        texts.append(
            chunk
        )

        metadatas.append({
            "source": file_path.name,
            "file_type": "txt",
            "chunk_number": chunk_number,
            "file_hash": file_hash,
            "uploaded": True
        })

        ids.append(
            f"upload_"
            f"{file_hash[:12]}_"
            f"txt_"
            f"{chunk_number}"
        )

    return (
        texts,
        metadatas,
        ids
    )


# --------------------------------------------------
# PDF CHUNKS
# --------------------------------------------------

def extract_pdf_chunks(
    file_path,
    file_hash
):

    reader = PdfReader(
        str(file_path)
    )

    texts = []
    metadatas = []
    ids = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = (
            page.extract_text()
            or ""
        )

        if not page_text.strip():

            continue

        page_chunks = (
            text_splitter.split_text(
                page_text
            )
        )

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1
        ):

            texts.append(
                chunk
            )

            metadatas.append({
                "source": file_path.name,
                "file_type": "pdf",
                "page": page_number,
                "chunk_number": chunk_number,
                "file_hash": file_hash,
                "uploaded": True
            })

            ids.append(
                f"upload_"
                f"{file_hash[:12]}_"
                f"pdf_"
                f"page_{page_number}_"
                f"chunk_{chunk_number}"
            )

    return (
        texts,
        metadatas,
        ids
    )


# --------------------------------------------------
# INDEX UPLOADED FILE
# --------------------------------------------------

def index_uploaded_file(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )

    file_hash = (
        create_file_hash(
            file_bytes
        )
    )

    source_name = (
        uploaded_file.name
    )

    suffix = (
        Path(source_name)
        .suffix
        .lower()
    )


    if file_already_indexed(
        file_hash
    ):

        return {
            "status": "already_exists",
            "file": source_name,
            "chunks": 0,
            "deleted": 0
        }


    file_path = (
        UPLOAD_FOLDER
        / source_name
    )

    file_path.write_bytes(
        file_bytes
    )


    if suffix == ".txt":

        (
            texts,
            metadatas,
            ids
        ) = extract_txt_chunks(
            file_path,
            file_hash
        )

    elif suffix == ".pdf":

        (
            texts,
            metadatas,
            ids
        ) = extract_pdf_chunks(
            file_path,
            file_hash
        )

    else:

        raise ValueError(
            "Only TXT and PDF files are supported."
        )


    if not texts:

        raise ValueError(
            "No readable text was found "
            "inside the uploaded file."
        )


    deleted_count = (
        delete_existing_source(
            source_name
        )
    )


    total_chunks = len(
        texts
    )

    progress_bar = (
        st.progress(0)
    )

    status_text = (
        st.empty()
    )

    added_ids = []


    try:

        for start_index in range(
            0,
            total_chunks,
            BATCH_SIZE
        ):

            end_index = min(
                start_index
                + BATCH_SIZE,
                total_chunks
            )

            batch_texts = (
                texts[
                    start_index:end_index
                ]
            )

            batch_metadatas = (
                metadatas[
                    start_index:end_index
                ]
            )

            batch_ids = (
                ids[
                    start_index:end_index
                ]
            )

            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            added_ids.extend(
                batch_ids
            )

            progress = (
                end_index
                / total_chunks
            )

            progress_bar.progress(
                progress
            )

            status_text.write(
                f"Indexed "
                f"{end_index}/"
                f"{total_chunks} chunks"
            )


    except Exception:

        if added_ids:

            vector_store.delete(
                ids=added_ids
            )

        raise


    progress_bar.empty()
    status_text.empty()


    return {
        "status": "indexed",
        "file": source_name,
        "chunks": total_chunks,
        "deleted": deleted_count
    }


# --------------------------------------------------
# GET INDEXED DOCUMENTS
# --------------------------------------------------

def get_indexed_documents():

    data = vector_store.get(
        include=[
            "metadatas"
        ]
    )

    metadatas = (
        data.get(
            "metadatas",
            []
        )
    )

    documents = {}


    for metadata in metadatas:

        if metadata is None:

            continue


        source = metadata.get(
            "source",
            "Unknown"
        )

        file_type = metadata.get(
            "file_type",
            "Unknown"
        )

        uploaded = metadata.get(
            "uploaded",
            False
        )


        if source not in documents:

            documents[source] = {
                "file_type": file_type,
                "uploaded": uploaded,
                "chunks": 0
            }


        documents[source][
            "chunks"
        ] += 1


    return documents


# --------------------------------------------------
# DISPLAY SOURCES
# --------------------------------------------------

def display_sources(
    source_list
):

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
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header(
        "📄 Knowledge Base"
    )

    st.write(
        "Upload a TXT or PDF file "
        "and add it directly to Chroma."
    )


    uploaded_file = (
        st.file_uploader(
            "Choose a document",
            type=[
                "txt",
                "pdf"
            ]
        )
    )


    add_document = (
        st.button(
            "Add to Knowledge Base",
            use_container_width=True
        )
    )


    if (
        add_document
        and uploaded_file
        is not None
    ):

        try:

            with st.spinner(
                "Processing and indexing document..."
            ):

                result = (
                    index_uploaded_file(
                        uploaded_file
                    )
                )


            if (
                result["status"]
                == "already_exists"
            ):

                st.info(
                    f"{result['file']} "
                    f"is already indexed."
                )

            else:

                st.success(
                    f"{result['file']} "
                    f"indexed successfully."
                )

                st.write(
                    f"Chunks added: "
                    f"{result['chunks']}"
                )

                if (
                    result["deleted"]
                    > 0
                ):

                    st.write(
                        "Previous version replaced: "
                        f"{result['deleted']} "
                        "old chunks removed."
                    )


        except Exception as error:

            st.error(
                f"Indexing failed: "
                f"{error}"
            )


    elif (
        add_document
        and uploaded_file
        is None
    ):

        st.warning(
            "Please choose a TXT "
            "or PDF file first."
        )


    # --------------------------------------------------
    # INDEXED DOCUMENTS
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "📚 Indexed Documents"
    )


    indexed_documents = (
        get_indexed_documents()
    )


    if not indexed_documents:

        st.write(
            "No documents indexed."
        )

    else:

        st.write(
            f"Total Documents: "
            f"{len(indexed_documents)}"
        )


        for source, details in sorted(
            indexed_documents.items()
        ):

            with st.expander(
                source
            ):

                st.write(
                    f"**Type:** "
                    f"{details['file_type']}"
                )

                st.write(
                    f"**Chunks:** "
                    f"{details['chunks']}"
                )


                if details["uploaded"]:

                    st.write(
                        "**Source:** "
                        "Uploaded from app"
                    )


                    delete_button = (
                        st.button(
                            f"Delete {source}",
                            key=f"delete_{source}",
                            use_container_width=True
                        )
                    )


                    if delete_button:

                        delete_result = (
                            delete_uploaded_document(
                                source
                            )
                        )


                        if (
                            delete_result["status"]
                            == "deleted"
                        ):

                            st.success(
                                f"{source} deleted."
                            )

                            st.write(
                                "Chunks removed: "
                                f"{delete_result['deleted_chunks']}"
                            )

                            st.rerun()


                        elif (
                            delete_result["status"]
                            == "protected"
                        ):

                            st.error(
                                "This document is protected "
                                "and cannot be deleted."
                            )


                        else:

                            st.warning(
                                "Document was not found."
                            )


                else:

                    st.write(
                        "**Source:** "
                        "Base knowledge"
                    )

                    st.caption(
                        "Protected document"
                    )


# --------------------------------------------------
# MAIN HEADER
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

if st.button(
    "Clear Chat"
):

    st.session_state.messages = []

    if os.path.exists(
        CHAT_HISTORY_FILE
    ):

        os.remove(
            CHAT_HISTORY_FILE
        )

    st.rerun()


# --------------------------------------------------
# DISPLAY SAVED CHAT
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
                message[
                    "sources"
                ]
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


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


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


        st.markdown(
            response
        )


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