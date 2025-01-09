# import os
#
# import wikipedia
# from dotenv import load_dotenv
# from langchain_community.vectorstores import UpstashVectorStore
# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import TokenTextSplitter
# from upstash_vector import Index
#
# load_dotenv()
#
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# UPSTASH_VECTOR_URL = os.environ.get('UPSTASH_VECTOR_URL')
# UPSTASH_VECTOR_TOKEN = os.environ.get('UPSTASH_VECTOR_TOKEN')
#
# # ny = wikipedia.page(title="new York City, New York")
# # r = wikipedia.search("Nepalgunj")
# # print(r)
#
# # dim -> 1536
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#
# # dim -> 3072
# # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#
# upstash_index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
#
# documents = []
# cities = ['New York City, New York', 'kathmandu']
# for city in cities:
#     wikipedia_page_result = wikipedia.page(title=city)
#     doc = Document(
#         page_content=wikipedia_page_result.content,
#         metadata={
#             "source": f"{wikipedia_page_result.url}",
#             "title": wikipedia_page_result.title,
#         },
#     )
#     documents.append(doc)
#
# store = UpstashVectorStore(embedding=embeddings, index_url=UPSTASH_VECTOR_URL, index_token=UPSTASH_VECTOR_TOKEN)
# text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name='gpt-4o', chunk_size=100, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# print("Length of docs.......", len(docs))
#
# inserted_vectors = store.add_documents(docs)  # stores into vector database
# print("Stored into vector database successfully...")
# result = store.similarity_search_with_score("The city of temples...", k=1)
# for doc, score in result:
#     print(f" Documents {doc}: Score {score}")
