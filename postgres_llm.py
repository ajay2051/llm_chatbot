import os

from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")

connection_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
print("Connection Successfully.....")

db = SQLDatabase.from_uri(connection_uri)
LLM_CONFIG = {
    "model": "gpt-3.5-turbo",
    "api_key": OPENAI_API_KEY,
    "temperature": 0
}
llm = ChatOpenAI(**LLM_CONFIG)
prompt_template = """
You are a helpful assistant designed to safely interact with a SQL database.
Your tasks include selecting data and answering questions about the database schema.

IMPORTANT: Do not modify, delete, or insert any data in the database unless explicitly asked to do so.
If you are unsure about the action, only query or return information from the database.

Available tools: {tools}
Available tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"], template=prompt_template,)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, prompt=prompt, verbose=True, handle_parsing_errors=True)

output = agent_executor.invoke({"input":"How many users are there"})
print("Output...", output)
