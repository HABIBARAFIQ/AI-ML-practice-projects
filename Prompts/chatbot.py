from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# 1. Keep the endpoint initialized (together provider expects conversational)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="conversational",
    temperature=0.9
)

# 2. Wrap it with ChatHuggingFace so it formats as a conversational chat payload
chat_model = ChatHuggingFace(llm=llm)
chat_history = []
while True:
    user_input = input("User: ")
    chat_history.append({"role": "user", "content": user_input})
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_model.invoke(chat_history)
    chat_history.append({"role": "assistant", "content": response.content})
    print(f"Assistant: {response.content}")


print("Chat session ended.")
print("Chat History:")
for message in chat_history:
    print(f"{message['role'].capitalize()}: {message['content']}")
        