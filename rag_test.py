from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# SETTINGS
# -----------------------------

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# -----------------------------
# VECTOR STORE
# -----------------------------

vector_store = InMemoryVectorStore(
    embedding=embeddings
)


# -----------------------------
# OUR KNOWLEDGE BASE
# -----------------------------

documents = [
    "Random Forest uses many decision trees. "
    "Each tree is trained on a random sample of the data and random features. "
    "For classification, the trees vote and the majority prediction becomes the final result.",

    "XGBoost is a boosting algorithm that builds decision trees sequentially. "
    "Each new tree tries to correct errors made by the previous trees.",

    "Gradient Descent is an optimization algorithm used to minimize a loss function.",

    "Overfitting happens when a model performs very well on training data "
    "but performs poorly on new unseen data.",

    "Pizza is a popular food made with dough, sauce, cheese, and toppings."
]


# Store documents as embeddings
vector_store.add_texts(documents)


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

        Answer the user's question using only the provided context.

        If the answer is not available in the context,
        say:

        "I do not have enough information in the provided context."

        Keep the answer simple and accurate.
        """
    ),
    (
        "human",
        """
        Context:

        {context}

        Question:

        {question}
        """
    )
])


# Convert AIMessage into plain text
output_parser = StrOutputParser()


# Create chain
chain = prompt | llm | output_parser


# -----------------------------
# USER QUESTION
# -----------------------------

question = "How does Random Forest work?"


# -----------------------------
# RETRIEVE RELEVANT DOCUMENTS
# -----------------------------

results = vector_store.similarity_search(
    question,
    k=2
)


# Combine retrieved documents
context = "\n\n".join(
    document.page_content
    for document in results
)


# -----------------------------
# SHOW RETRIEVED CONTEXT
# -----------------------------

print("Question:")
print(question)

print("\nRetrieved Context:")
print(context)


# -----------------------------
# GENERATE RAG ANSWER
# -----------------------------

response = chain.invoke({
    "context": context,
    "question": question
})


print("\nRAG Answer:")
print(response)