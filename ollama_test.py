import json
import os
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"
HISTORY_FILE = "chat_history.json"


SYSTEM_PROMPT = """
You are a beginner-friendly AI Learning Assistant.

Your responsibilities:
- Explain technical topics in simple words.
- Teach step by step.
- Use short examples when useful.
- Avoid unnecessary complexity.
- Answer clearly and concisely.
"""


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_history(conversation):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            conversation,
            file,
            indent=4,
            ensure_ascii=False
        )


conversation = load_history()


def chat_with_ollama(user_message):
    conversation.append({
        "role": "user",
        "content": user_message
    })

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ] + conversation

    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }

    json_data = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    assistant_message = result["message"]["content"]

    conversation.append({
        "role": "assistant",
        "content": assistant_message
    })

    save_history(conversation)

    return assistant_message


print("Customized Local Ollama Chatbot")
print("Model:", MODEL_NAME)
print("Saved messages:", len(conversation))

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
        conversation.clear()

        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)

        print("\nBot: Chat history cleared.\n")
        continue

    response = chat_with_ollama(user_input)

    print("\nBot:")
    print(response)
    print()