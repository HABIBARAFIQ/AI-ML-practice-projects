from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, TypedDict, Annotated, Literal
from dotenv import load_dotenv
# Movie title
# Overall sentiment
# Numeric rating
# Positive aspects
# Negative aspects
# Recommended audience
# Genres
# Confidence score
load_dotenv()
class MovieReview(BaseModel):
    title: str = Field(..., description="The title of the movie")
    sentiment: Annotated[str, Literal["positive", "negative", "neutral"]] = Field(..., description="Overall sentiment of the review")
    rating: float = Field(..., ge=0, le=10, description="Numeric rating of the movie (0-10)")
    positive_aspects: Optional[list[str]] = None
    negative_aspects: Optional[list[str]] = None
    recommended_audience: str = Field(..., description="Recommended audience for the movie")
    genres: list[str] = Field(..., description="Genres associated with the movie")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score of the analysis (0-1)")
    recommended: bool = Field(..., description="Whether the movie is recommended or not")

# 1. Keep the endpoint initialized (together provider expects conversational)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation",
)
chat_model = ChatHuggingFace(llm=llm)
#1st prompt ->Detailed report
parser = PydanticOutputParser(pydantic_object=MovieReview)
template1 = PromptTemplate(
    template = "You are a movie review analyzer. Analyze the following movie review and provide a structured output in JSON format with the following fields: title, sentiment, rating, positive_aspects, negative_aspects, recommended_audience, genres, confidence_score, recommended.{format_instructions} Review: {review}",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)
# parser die shb metadata fele dei only result rekhe dei 
chain = template1 | chat_model | parser
result = chain.invoke({"review": "I watched Interstellar yesterday. The cinematography was breathtaking and the soundtrack was phenomenal. Although the first half felt a little slow, the ending was outstanding. I would rate it 9.5/10 and highly recommend it to science fiction lovers."})
print(result.model_dump_json(indent=4))