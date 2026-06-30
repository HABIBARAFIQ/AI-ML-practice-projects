from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a experienced {profession}."),
     ("user", "Tell me about {topic}.")])
print(prompt.format_messages(topic="viral infection", profession="doctor"))