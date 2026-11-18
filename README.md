# Local RAG Assistant with Ollama, LangChain, Chroma and Streamlit

A fully local Retrieval-Augmented Generation (RAG) application built with Python, Ollama, LangChain, Chroma and Streamlit.

The application allows users to upload TXT and PDF documents, index them locally, ask questions from the knowledge base and receive grounded answers with source traceability.

## Features

- Local LLM using Ollama
- `llama3.2:3b` for answer generation
- `nomic-embed-text` for embeddings
- LangChain prompt pipelines
- Persistent Chroma vector database
- Streamlit chat interface
- TXT and PDF document support
- Automatic document chunking
- Batch embedding for large documents
- Dynamic file upload from the browser
- Automatic Chroma indexing
- Duplicate document detection
- Updated-file replacement
- Indexed-document listing
- Safe deletion of uploaded documents
- Protected base knowledge documents
- Semantic similarity retrieval
- Dynamic retrieval filtering
- Source file tracking
- PDF page tracking
- Chunk tracking
- Retrieval distance display
- Conversational follow-up handling
- Persistent chat history
- Unsupported-question guardrail

## Project Architecture

```text
TXT / PDF Documents
        ↓
Text Extraction
        ↓
Recursive Chunking
        ↓
nomic-embed-text
        ↓
Chroma Vector Database
        ↓
User Question
        ↓
Semantic Retrieval
        ↓
Distance Filtering
        ↓
Relevant Context
        ↓
llama3.2:3b
        ↓
Grounded Answer
        ↓
Source + Page + Chunk Traceability
```

## Streamlit Workflow

```text
Upload Document
      ↓
Save Locally
      ↓
Extract Text
      ↓
Create Chunks
      ↓
Generate Embeddings in Batches
      ↓
Add to Chroma
      ↓
Ask Questions Immediately
```

No manual database rebuild is required after uploading a document through the application.

## Project Structure

```text
ollama-local-chatbot/
│
├── knowledge/
│   ├── dl_knowledge.txt
│   ├── ml_knowledge.txt
│   ├── sample_knowledge.pdf
│   │
│   └── uploads/
│       └── runtime uploaded files
│
├── app.py
├── build_chroma_db.py
├── chroma_query_test.py
├── embedding_test.py
├── langchain_ollama_test.py
├