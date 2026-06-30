from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = "gemini-3.5-flash"
response = ChatGoogleGenerativeAI(model = model, temperature = 0.9).invoke("What is the capital of France?")
print(response.content)