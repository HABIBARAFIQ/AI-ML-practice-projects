from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Flash", 
                          task="text-generation",
                          temperature=0.9)
response = ChatHuggingFace(llm=llm).invoke("How beautiful is the sky?")
print(response.content)