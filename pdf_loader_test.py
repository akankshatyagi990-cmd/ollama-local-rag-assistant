from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# -----------------------------
# SETTINGS
# -----------------------------

PDF_FILE = "knowledge/sample_knowledge.pdf"


# -----------------------------
# CHECK PDF
# -----------------------------

pdf_path = Path(PDF_FILE)

if not pdf_path.exists():
    print("Error: PDF file not found.")
    exit()


# -----------------------------
# LOAD PDF
# -----------------------------

loader = PyPDFLoader(
    PDF_FILE
)

documents = loader.load()


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("PDF Loaded Successfully")

print("\nTotal Pages:")
print(len(documents))


for index, document in enumerate(
    documents,
    start=1
):

    print(
        f"\n---------- Page {index} ----------"
    )

    print(
        document.page_content[:500]
    )

    print("\nMetadata:")
    print(document.metadata)