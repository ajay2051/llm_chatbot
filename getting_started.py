import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

LLM_CONFIG = {
    "api_key": OPENAI_API_KEY,
    "model": "gpt-3.5-turbo",
    "temperature":0,
    "max_tokens":None,
    "timeout":None,
    "max_retries":2
}
model = ChatOpenAI(**LLM_CONFIG)

message = [HumanMessage(content="hello"), SystemMessage(content="Translate from english to nepali")]
result = model.invoke(message)
print(result.content)
parser = StrOutputParser()
print(parser.invoke(result))
