from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = "gpt-4.0-turbo"
response = ChatOpenAI(model = model, temperature = 0.9).invoke("What is the capital of France?")
print(response.content)
