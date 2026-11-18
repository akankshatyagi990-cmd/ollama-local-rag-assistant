# Local RAG Assistant with Ollama, LangChain and Chroma

A fully local Retrieval-Augmented Generation (RAG) project built with Python, Ollama, LangChain and Chroma.

The assistant can answer questions from local TXT and PDF documents without using a cloud LLM.

## Features

- Local LLM using Ollama
- `llama3.2:3b` for answer generation
- `nomic-embed-text` for embeddings
- LangChain prompt pipelines
- Persistent Chroma vector database
- TXT document support
- PDF document support using `pypdf`
- Automatic document chunking
- Semantic similarity search
- Retrieval distance filtering
- Source file tracking
- PDF page number tracking
- Conversational follow-up questions
- Persistent chat history
- Unsupported-question guardrail

## Project Structure

```text
ollama-local-chatbot/
│
├── knowledge/
│   ├── ml_knowledge.txt
│   ├── dl_knowledge.txt
│   └── sample_knowledge.pdf
│
├── build_chroma_db.py
├── rag_file_test.py
├── pdf_rag_test.py
├── pdf_distance_test.py
├── multi_source_distance_test.py
├── embedding_test.py
├── vector_store_test.py
├── chroma_query_test.py
├── langchain_ollama_test.py
├── ollama_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Models Used

### LLM

```text
llama3.2:3b
```

Used to generate answers locally.

### Embedding Model

```text
nomic-embed-text
```

Used to convert document chunks and user questions into embedding vectors.

## RAG Workflow

```text
TXT / PDF Documents
        ↓
Text Extraction
        ↓
Chunking
        ↓
nomic-embed-text
        ↓
Embeddings
        ↓
Chroma Vector Database
        ↓
User Question
        ↓
Semantic Retrieval
        ↓
Relevant Context
        ↓
llama3.2:3b
        ↓
Grounded Answer
```

## Installation

Install Ollama and make sure it is running.

Pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Build the Vector Database

Place TXT and PDF documents inside the `knowledge` folder.

Then run:

```bash
python build_chroma_db.py
```

This extracts text, splits it into chunks, generates embeddings and stores them inside a persistent Chroma database.

## Run the RAG Assistant

```bash
python rag_file_test.py
```

Example:

```text
You: What does Gradient Descent do?

Bot:
Gradient Descent minimizes a loss function by repeatedly adjusting model parameters.
```

The assistant also displays the retrieved source file, chunk number, PDF page number when applicable, and retrieval distance.

## Guardrails

The application filters weak retrieval results before sending context to the LLM.

If relevant information is not available in the knowledge base, the assistant responds:

```text
I do not have enough information in the provided documents.
```

This helps reduce unsupported or hallucinated answers.

## Local Privacy

The LLM, embeddings and vector database run locally through Ollama and Chroma.

No cloud LLM API key is required for this project.