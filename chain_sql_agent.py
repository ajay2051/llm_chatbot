import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASS")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")

# Create database connection URI
connection_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
db = SQLDatabase.from_uri(connection_uri)

# LLM configuration
LLM_CONFIG = {
    "model": "gpt-4o",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,  # Slightly higher temperature for more creative responses
    "max_tokens": 2000
}
llm = ChatOpenAI(**LLM_CONFIG)

# Define the database schema information
DATABASE_SCHEMA = """
Here is the schema of the database:

1. **b_product**: Products
   - Columns: product_id (Primary Key), product_name

2. **b_package**: Package types
   - Columns: package_id (Primary Key), package_type

3. **b_package_size**: Package sizes
   - Columns: size_id (Primary Key), package_size, count

4. **b_prod_pack_map**: Product-package-size relationships
   - Columns: product_id (Foreign Key), package_id (Foreign Key), size_id (Foreign Key)

5. **b_qr_code**: QR code data
   - Columns: qr_code_id (Primary Key), unique_qr_id, product_id (Foreign Key), package_id (Foreign Key), size_id (Foreign Key), package_date, 

6. **b_pallete_move**: Palette movement history
   - Columns: pallette_move_id (Primary Key), unique_qr_id (Foreign Key), w_aile, w_row, w_level, status, move_date

7. **b_active_palettes**: Current palette status and location
   - Columns: id (Primary Key), unique_qr_id (Foreign Key),  w_aile, w_row, w_level, l_status, updated_dt
"""

# Define the system role for the AI
AGENT_LLM_SYSTEM_ROLE = f"""
You are a friendly and knowledgeable AI assistant. Your task is to help users query a database and provide clear, concise, and conversational answers based on the results.

Here is the schema of the database:
{DATABASE_SCHEMA}

Rules:
1. Always provide a human-readable response.
2. If the query result is a number or a list, explain it in a conversational way.
3. If the query result is empty, let the user know politely.
4. If there's an error, explain it in simple terms and suggest what the user can do next.

Example:
- User: "How many products are there?"
- AI: "There are 150 products in the warehouse. Let me know if you'd like more details!"
"""

# Initialize tools and chains

execute_query = QuerySQLDatabaseTool(db=db) # Tool to execute SQL queries

# Define the prompt for SQL generation
sql_prompt = PromptTemplate.from_template(
    """
    You are a SQL expert. Given the following database schema:

    {schema}

    Generate a valid SQL query to answer the following question:
    {question}

    Return only the SQL query, nothing else.
    """
)

# Chain to generate SQL queries
write_query = (
    RunnablePassthrough.assign(schema=lambda x: DATABASE_SCHEMA)
    | sql_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().strip("```sql").strip("```"))  # Strip markdown and extra whitespace
)

# Define the answer prompt
answer_prompt = PromptTemplate.from_template(
    """
    You are a friendly AI assistant. Below is the result of a database query:

    Query Result: {result}

    Based on this result, provide a clear and conversational answer to the user's question: {question}
    """
)

# Define the answer chain
answer = answer_prompt | llm | StrOutputParser()

# Define the full chain
chain = (
    RunnablePassthrough.assign(query=write_query)  # Generate SQL query
    .assign(result=itemgetter("query") | execute_query)  # Execute SQL query
    | answer  # Provide a conversational answer
)

# Function to interact with the agent
def ask_agent(question: str) -> str:
    try:
        # Generate the SQL query
        query = write_query.invoke({"question": question}).strip()
        print(f"Generated Query: {query}")  # Debugging: log the query

        # Execute the query
        result = execute_query.invoke(query)
        return answer.invoke({"question": question, "result": result})
    except Exception as e:
        return f"Query error: {str(e)}. Please verify your question and database schema."


# Test the agent
if __name__ == "__main__":
    # Example questions
    test_questions = [
        "List all products",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = ask_agent(question)
        print(f"Response: {response}")