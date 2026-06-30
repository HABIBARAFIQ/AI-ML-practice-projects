from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=model,dimensions=1536)
document = ["What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Italy?"]

response = embeddings.embed_documents(document)
print(str(response))