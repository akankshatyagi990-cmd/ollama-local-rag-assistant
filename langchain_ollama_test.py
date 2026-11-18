from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from langchain_core.output_parsers import StrOutputParser


MODEL_NAME = "llama3.2:3b"


# Local Ollama model
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1
)


# Reusable prompt with chat history
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a beginner-friendly AI Learning Assistant.

        Follow these rules:
        - Explain technical concepts in simple words.
        - Keep answers concise.
        - Be technically accurate.
        - Use one simple example when useful.
        - Do not provide code unless requested.
        - If unsure, clearly say you are unsure.
        """
    ),

    # Previous conversation will be inserted here
    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    (
        "human",
        "{question}"
    )
])


output_parser = StrOutputParser()


# Build chain
chain = prompt | llm | output_parser


# Store conversation in memory
chat_history = []


print("LangChain + Ollama Learning Assistant")
print("Model:", MODEL_NAME)

print("\nCommands:")
print("exit / quit")
print("clear chat")
print()


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    if user_input.lower() == "clear chat":

        chat_history.clear()

        print("\nBot: Conversation memory cleared.\n")
        continue


    # Send question + previous conversation
    response = chain.invoke({
        "question": user_input,
        "chat_history": chat_history
    })


    # Save user message
    chat_history.append(
        HumanMessage(
            content=user_input
        )
    )


    # Save assistant response
    chat_history.append(
        AIMessage(
            content=response
        )
    )


    print("\nBot:")
    print(response)
    print()