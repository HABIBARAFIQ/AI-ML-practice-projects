from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
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

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = chat_model.invoke(messages)
ans =AIMessage(content=response.content)
print(ans.content)