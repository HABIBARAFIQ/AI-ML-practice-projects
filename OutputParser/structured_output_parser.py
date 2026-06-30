# Now its deprecated ,dont need this code
# from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()


# 1. Keep the endpoint initialized (together provider expects conversational)
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", 
    task="text-generation",
    temperature=0.9
)
chat_model = ChatHuggingFace(llm=llm)
parser = StructuredOutputParser.from_response_schemas(
    response_schemas=[
        ResponseSchema(name="fact-1", description="A fact about the topic."),
        ResponseSchema(name="fact-2", description="Another fact about the topic."),
        ResponseSchema(name="fact-3", description="A third fact about the topic.")
    ]   
)
template1 = PromptTemplate(
    template="You are a helpful assistant.{format_instructions} Provide 3 facts about {topic}.",
    input_variables=["topic"],
     partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)
chain = template1 | chat_model | parser
result = chain.invoke({"topic":"Black holes"})
print(result)