from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
text = "What is the capital of France?"
response = embeddings.embed_query(text)

document = ["What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Italy?"]
document_embeddings = embeddings.embed_documents(document)
similarity_scores = cosine_similarity([response], document_embeddings)
# print("Similarity scores:", similarity_scores)
result = sorted(list(enumerate(similarity_scores[0])), key=lambda x: x[1],reverse=True)[0]
print(f"The most similar document to the query is: '{document[result[0]]}' with a similarity score of {result[1]:.4f}")