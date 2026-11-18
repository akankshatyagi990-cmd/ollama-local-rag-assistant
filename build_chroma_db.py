import shutil
from pathlib import Path

from pypdf import PdfReader

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

KNOWLEDGE_FOLDER = Path("knowledge")

PERSIST_DIRECTORY = "chroma_db"

COLLECTION_NAME = "local_rag"

EMBEDDING_MODEL = "nomic-embed-text"

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
# STORAGE
# --------------------------------------------------

all_chunks = []

all_metadatas = []

all_ids = []

txt_file_count = 0

pdf_file_count = 0


# --------------------------------------------------
# CHECK KNOWLEDGE FOLDER
# --------------------------------------------------

if not KNOWLEDGE_FOLDER.exists():

    print(
        "Knowledge folder does not exist."
    )

    raise SystemExit


# --------------------------------------------------
# DISCOVER FILES
# --------------------------------------------------

knowledge_files = sorted(
    list(
        KNOWLEDGE_FOLDER.glob("*.txt")
    )
    +
    list(
        KNOWLEDGE_FOLDER.glob("*.pdf")
    )
)


if not knowledge_files:

    print(
        "No TXT or PDF files found "
        "inside the knowledge folder."
    )

    raise SystemExit


print(
    "Loading Knowledge Files:\n"
)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

for file_path in knowledge_files:

    suffix = (
        file_path.suffix.lower()
    )


    # --------------------------------------------------
    # TXT FILE
    # --------------------------------------------------

    if suffix == ".txt":

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = (
            text_splitter.split_text(
                text
            )
        )

        txt_file_count += 1


        print(
            f"TXT: {file_path.name} "
            f"→ {len(chunks)} chunks"
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
                f"txt_{file_path.stem}_"
                f"{chunk_number}"
            )


    # --------------------------------------------------
    # PDF FILE
    # --------------------------------------------------

    elif suffix == ".pdf":

        reader = PdfReader(
            str(file_path)
        )

        pdf_file_count += 1

        total_pdf_chunks = 0


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

                total_pdf_chunks += 1


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
                    f"pdf_{file_path.stem}_"
                    f"page_{page_number}_"
                    f"chunk_{chunk_number}"
                )


        print(
            f"PDF: {file_path.name} "
            f"→ {len(reader.pages)} pages "
            f"→ {total_pdf_chunks} chunks"
        )


# --------------------------------------------------
# CHECK CHUNKS
# --------------------------------------------------

if not all_chunks:

    print(
        "\nNo text could be extracted "
        "from the knowledge files."
    )

    raise SystemExit


# --------------------------------------------------
# REMOVE OLD DATABASE
# --------------------------------------------------

persist_path = Path(
    PERSIST_DIRECTORY
)


if persist_path.exists():

    shutil.rmtree(
        persist_path
    )

    print(
        "\nOld Chroma database removed."
    )


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# --------------------------------------------------
# CREATE EMPTY CHROMA DATABASE
# --------------------------------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)


# --------------------------------------------------
# ADD CHUNKS IN SMALL BATCHES
# --------------------------------------------------

total_chunks = len(
    all_chunks
)

print(
    "\nCreating embeddings in batches..."
)

print(
    f"Batch Size: {BATCH_SIZE}"
)

print(
    f"Total Chunks: {total_chunks}\n"
)


for start_index in range(
    0,
    total_chunks,
    BATCH_SIZE
):

    end_index = min(
        start_index + BATCH_SIZE,
        total_chunks
    )


    batch_chunks = (
        all_chunks[
            start_index:end_index
        ]
    )


    batch_metadatas = (
        all_metadatas[
            start_index:end_index
        ]
    )


    batch_ids = (
        all_ids[
            start_index:end_index
        ]
    )


    vector_store.add_texts(
        texts=batch_chunks,
        metadatas=batch_metadatas,
        ids=batch_ids
    )


    print(
        f"Embedded: "
        f"{end_index}/{total_chunks}"
    )


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print(
    "\nChroma Database Created Successfully"
)

print(
    f"Total Files: "
    f"{txt_file_count + pdf_file_count}"
)

print(
    f"Total Chunks: "
    f"{total_chunks}"
)

print(
    f"TXT Files: "
    f"{txt_file_count}"
)

print(
    f"PDF Files: "
    f"{pdf_file_count}"
)

print(
    f"Database Folder: "
    f"{PERSIST_DIRECTORY}"
)

print(
    f"Collection: "
    f"{COLLECTION_NAME}"
)