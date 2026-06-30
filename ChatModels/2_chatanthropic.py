from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()
model = "claude-3.5-sonnet"
response = ChatAnthropic(model = model, temperature = 0.9).invoke("What is the capital of France?")
print(response.content)