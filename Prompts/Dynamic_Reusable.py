from langchain_core.prompts import PromptTemplate,ChatPromptTemplate

prompt = PromptTemplate.from_template("Tell me about {topic} in {emotion} tone.")
print(prompt.format(topic="the Eiffel Tower", emotion="excited"))

# In different way
prompt2 = ChatPromptTemplate([
    ("system", "You are a {domain} specialist."),
    ("user", "Tell me about {topic}.")
])
print(prompt2.invoke({"domain": "Python", "topic": "functions"}))