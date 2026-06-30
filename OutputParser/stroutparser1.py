# from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


# 1. Keep the endpoint initialized (together provider expects conversational)
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", 
    task="text-generation",
    temperature=0.9
)
chat_model = ChatHuggingFace(llm=llm)
#1st prompt ->Detailed report
template1 = PromptTemplate(
    template="You are a helpful assistant. Provide a detailed report on {topic}.",
    input_variables=["topic"]
)
#2nd prompt ->Summary
template2 = PromptTemplate(
    template="You are a helpful assistant. Provide a 5 line summary of the following text: {text}.",
    input_variables=["text"]
)
parser = StrOutputParser()
# parser die shb metadata fele dei only result rekhe dei 
chain = template1 | chat_model | parser | template2 | chat_model| parser
result = chain.invoke({"topic":"Black holes"})
print(result)