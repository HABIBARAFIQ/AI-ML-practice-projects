from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
model = "text-embedding-3-small"
response = OpenAIEmbeddings(model=model,dimensions=1536).embed_query("What is the capital of France?")
print(str(response))