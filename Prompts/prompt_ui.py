import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt

# Load environment variables
# env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv()

# 1. Keep the endpoint initialized (together provider expects conversational)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="conversational",
    temperature=0.9
)

# 2. Wrap it with ChatHuggingFace so it formats as a conversational chat payload
chat_model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name", 
    [
        "Attention Is All You Need", 
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
) 

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('E:/Langchain/Prompts/template.json')

if st.button('Summarize'):
    # Chain using the chat_model instead of raw llm
    chain = template | chat_model
    
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    
    # Since we are using chat_model, result.content is now completely valid!
    st.write(result.content)