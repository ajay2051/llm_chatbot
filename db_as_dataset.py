import os
from typing import List

import psycopg2
from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from psycopg2.extras import RealDictCursor

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
UPSTASH_VECTOR_URL = os.environ.get('UPSTASH_VECTOR_URL')
UPSTASH_VECTOR_TOKEN = os.environ.get('UPSTASH_VECTOR_TOKEN')

# PostgreSQL connection settings
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT')


def get_db_connection():
    """Create and return a database connection"""
    db = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        cursor_factory=RealDictCursor
    )
    return db


def fetch_all_data() -> List[Document]:
    """Fetch cities data from PostgreSQL and convert to Documents"""
    documents = []

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Query 1: Fetch users data
            cur.execute("""
                SELECT 
                    username,
                    firstname,
                    lastname,
                    phoneno,
                    emailid,
                    last_login,
                    is_deleted
                FROM users
                WHERE username IS NOT NULL  -- Filter out NULL usernames
            """)

            for row in cur.fetchall():
                # Convert any None values to empty strings for metadata
                metadata = {
                    "first_name": row['firstname'] or "",
                    "last_name": row['lastname'] or "",
                    "phoneno": row['phoneno'] or "",
                    "emailid": row['emailid'] or "",
                    "last_login": str(row['last_login']) if row['last_login'] else "",
                    "is_deleted": bool(row['is_deleted'])
                }

                doc = Document(
                    page_content=str(row['username']),  # Convert to string to ensure non-null
                    metadata=metadata
                )
                documents.append(doc)

            # Query 2: Fetch roles
            cur.execute("""
                SELECT 
                    rolename
                FROM roles
                WHERE rolename IS NOT NULL  -- Filter out NULL rolenames
            """)
            for row in cur.fetchall():
                # Convert None to empty string if necessary
                rolename = str(row['rolename']) if row['rolename'] else "undefined_role"
                doc = Document(
                    page_content=rolename,
                    metadata={}
                )
                documents.append(doc)

    finally:
        conn.close()
    return documents


# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Fetch documents from PostgreSQL
documents = fetch_all_data()

# Initialize vector store
store = UpstashVectorStore(embedding=embeddings, index_url=UPSTASH_VECTOR_URL, index_token=UPSTASH_VECTOR_TOKEN)

# Split documents
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name='gpt-4', chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print("Length of docs.......", len(docs))

# Store in vector database
inserted_vectors = store.add_documents(docs)
print("Stored postgres db data into vector database successfully...")

# Search example
# result = store.similarity_search_with_score("The city of temples...", k=1)
# for doc, score in result:
#     print(f"Document: {doc}\nScore: {score}\n")