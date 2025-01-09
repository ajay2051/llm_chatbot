import os
from typing import Dict

from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from upstash_vector import Index

from db_as_dataset import get_db_connection

load_dotenv()


def get_schema():
    """Return the database schema description"""
    schema = {}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Query for tables and columns
            cur.execute("""
                    SELECT table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                """)
            for row in cur.fetchall():
                table = row["table_name"]
                column = row["column_name"]
                data_type = row["data_type"]
                if table not in schema:
                    schema[table] = []
                schema[table].append({"column_name": column, "data_type": data_type})
    finally:
        conn.close()
    return schema

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
UPSTASH_VECTOR_URL = os.environ.get('UPSTASH_VECTOR_URL')
UPSTASH_VECTOR_TOKEN = os.environ.get('UPSTASH_VECTOR_TOKEN')

# ny = wikipedia.page(title="new York City, New York")
# r = wikipedia.search("Nepalgunj")
# print(r)

# dim -> 1536
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# dim -> 3072
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

upstash_index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)

store = UpstashVectorStore(embedding=embeddings, index_url=UPSTASH_VECTOR_URL, index_token=UPSTASH_VECTOR_TOKEN)

# Chatbot
retriever = store.as_retriever(search_type='similarity', search_kwargs={"k": 1})

LLM_CONFIG = {
    "model": "gpt-4o",
    "api_key": OPENAI_API_KEY,
}

llm = ChatOpenAI(**LLM_CONFIG)

message = """
Answer this question: {question}

Database Schema Information:
{schema}

Context from similar documents:
{context}

Please provide an answer based on both the schema information and the context provided.
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])
# prompt = ChatPromptTemplate.from_template(template=message)

runnable = RunnableParallel( passed=RunnablePassthrough(), modified=lambda x: x["num"] + 1,)
runnable.invoke({"num": 1})

parser = StrOutputParser()


def get_response(query):
    try:
        chain = {"context": retriever, "question": RunnablePassthrough(), "schema": lambda _: get_schema()} | prompt | llm | parser
        response = chain.invoke({"question": query})
        return str(response)
    except Exception as e:
        print(e)
