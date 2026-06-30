from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
text = "What is the capital of France?"
response = embeddings.embed_query(text)
print(str(response))