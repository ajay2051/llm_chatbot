import os

#
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from upstash_vector import Index, Vector

#
load_dotenv()

phrase_1 = "The dog ate my homework"
phrase_2 = "The homework ate my dog"

phrase_1_as_list = sorted([x.lower() for x in phrase_1.split(" ")])
phrase_2_as_list = sorted([x.lower() for x in phrase_2.split(" ")])
print(phrase_1_as_list)
print(phrase_2_as_list)
print(phrase_1_as_list == phrase_2_as_list)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

LLM_CONFIG = {
    "api_key": OPENAI_API_KEY,
    "model": "gpt-3.5-turbo"
}
model = ChatOpenAI(**LLM_CONFIG)
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text, model="text-embedding-3-small"):
    """
    This function is used to get the embeddings of text.
    :param text: user_input_text
    :param model: gpt-3.5-turbo
    :return: embedding
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


documents = [
    "The cat jumped over the dog",
    "The cow jumped over the moon",
    "The turkey ran in circles"
]

embeddings = [get_embedding(x) for x in documents]
print(embeddings[0])
print(np.array(embeddings[0]).shape)


def calculate_cosine_metrics(v1, v2):
    """
    This function is used to calculate the cosine similarity between two vectors.
    :param v1: any_text
    :param v2: any_text
    :return: cosine_similarity
    """
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    cosine_distance = 1 - cosine_similarity
    return int(cosine_similarity * 100), int(cosine_distance * 100)


query_str = "The moose sat by turkey"
query_embedding = get_embedding(query_str)
for embedding in embeddings:
    print(calculate_cosine_metrics(query_embedding, embedding))

phrase_1_sorted = " ".join(phrase_1_as_list)
phrase_2_sorted = " ".join(phrase_2_as_list)

phrase_1_embedding = get_embedding(phrase_1_sorted)
phrase_2_embedding = get_embedding(phrase_2_sorted)
print(calculate_cosine_metrics(phrase_1_embedding, phrase_2_embedding))

# Upstash Vector
index = Index(url=os.environ.get("UPSTASH_VECTOR_URL"), token=os.environ.get("UPSTASH_VECTOR_TOKEN"))

dataset = {}
for i, embedding in enumerate(embeddings):
    dataset[i] = embedding

vectors = []
for key, value in dataset.items():
    print(key, value)
    my_id = key
    vectors.append(Vector(id=my_id, vector=value))
index.upsert(vectors=vectors)

results = index.query(vector=query_embedding, top_k=1, include_vectors=True, include_metadata=True)
for result in results:
    print("id", result.id, result.score, result.metadata)
