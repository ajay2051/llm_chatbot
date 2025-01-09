import requests

endpoint = "http://localhost:8000/langchain"
chain_type = "/invoke"
url = f"{endpoint}{chain_type}"
print(url)

data = {"input": "Tell me about lord ram"}
r = requests.post(url, json=data)
print(r.json())