from operator import pos

from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Optional, TypedDict, Annotated, Literal

from torch import neg
load_dotenv()

# 1. Keep the endpoint initialized (together provider expects conversational)
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", 
    task="text-generation",
    temperature=0.9
)
class Review(TypedDict):
    key_themes : Annotated[list[str], "The main themes or topics discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "The sentiment of the review (positive, negative, or neutral)"]
    pros: Annotated[Optional[list[str]], "The positive aspects of the movie"]
    cons: Annotated[Optional[list[str]], "The negative aspects of the movie"]
    name: Annotated[str, "The name of the movie being reviewed"]
# 2. Wrap it with ChatHuggingFace so it formats as a conversational chat payload
chat_model = ChatHuggingFace(llm=llm)
chat_model = chat_model.with_structured_output(Review)
result = chat_model.invoke("I watched Dune: Part Two last night, and it exceeded my expectations. The cinematography was breathtaking, with stunning desert landscapes and incredible visual effects that made every scene feel immersive. The performances were outstanding, especially from Timothée Chalamet and Zendaya, who brought emotional depth to their characters. Hans Zimmer's soundtrack perfectly complemented the intense atmosphere of the film. That said, the movie isn't without flaws. Some scenes felt a bit too slow, and viewers who haven't seen the first film may struggle to follow the complex political and cultural themes. A few supporting characters also deserved more screen time. Overall, it's an ambitious and visually spectacular science-fiction epic that rewards patient viewers with an engaging story and memorable performances. I would rate it 4.5 out of 5 stars and would highly recommend it to fans of science fiction and large-scale cinematic experiences.")
# print(result)
print(result['key_themes'])
print(result['summary'])
print(result['sentiment'])
print(result['pros'])
print(result['cons'])
print(result['name'])
