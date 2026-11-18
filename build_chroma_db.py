import shutil
from pathlib import Path

from pypdf import PdfReader

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# SETTINGS
# -----------------------------

KNOWLEDGE_FOLDER = "knowledge"

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "local_rag"

EMBEDDING_MODEL = "nomic-embed-text"


# -----------------------------
# KNOWLEDGE FOLDER
# -----------------------------

knowledge_path = Path(KNOWLEDGE_FOLDER)

if not knowledge_path.exists():
    print("Error: Knowledge folder not found.")
    exit()


# -----------------------------
# TEXT SPLITTER
# -----------------------------

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


# -----------------------------
# PREPARE DATA
# -----------------------------

all_chunks = []
all_metadatas = []
all_ids = []

total_files = 0


print("Loading Knowledge Files:\n")


# -----------------------------
# LOAD TXT FILES
# -----------------------------

txt_files = list(
    knowledge_path.glob("*.txt")
)


for file_path in txt_files:

    text = file_path.read_text(
        encoding="utf-8"
    )

    chunks = text_splitter.split_text(
        text
    )

    total_files += 1

    print(
        f"TXT: {file_path.name} → {len(chunks)} chunks"
    )

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        all_chunks.append(
            chunk
        )

        all_metadatas.append({
            "source": file_path.name,
            "file_type": "txt",
            "chunk_number": chunk_number
        })

        all_ids.append(
            f"{file_path.stem}_txt_chunk_{chunk_number}"
        )


# -----------------------------
# LOAD PDF FILES
# -----------------------------

pdf_files = list(
    knowledge_path.glob("*.pdf")
)


for file_path in pdf_files:

    reader = PdfReader(
        str(file_path)
    )

    total_files += 1

    pdf_chunk_count = 0


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        page_chunks = text_splitter.split_text(
            page_text
        )


        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1
        ):

            pdf_chunk_count += 1

            all_chunks.append(
                chunk
            )

            all_metadatas.append({
                "source": file_path.name,
                "file_type": "pdf",
                "page": page_number,
                "chunk_number": chunk_number
            })

            all_ids.append(
                f"{file_path.stem}_pdf_page_{page_number}_chunk_{chunk_number}"
            )


    print(
        f"PDF: {file_path.name} → "
        f"{len(reader.pages)} pages → "
        f"{pdf_chunk_count} chunks"
    )


# -----------------------------
# CHECK CONTENT
# -----------------------------

if not all_chunks:
    print(
        "\nError: No supported knowledge files found."
    )
    exit()


# -----------------------------
# REMOVE OLD DATABASE
# -----------------------------

database_path = Path(
    PERSIST_DIRECTORY
)

if database_path.exists():

    shutil.rmtree(
        database_path
    )

    print(
        "\nOld Chroma database removed."
    )


# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# -----------------------------
# CREATE CHROMA DATABASE
# -----------------------------

vector_store = Chroma.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    metadatas=all_metadatas,
    ids=all_ids,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)


# -----------------------------
# COMPLETE
# -----------------------------

print(
    "\nChroma Database Created Successfully"
)

print(
    "Total Files:",
    total_files
)

print(
    "Total Chunks:",
    len(all_chunks)
)

print(
    "TXT Files:",
    len(txt_files)
)

print(
    "PDF Files:",
    len(pdf_files)
)

print(
    "Database Folder:",
    PERSIST_DIRECTORY
)

print(
    "Collection:",
    COLLECTION_NAME
)