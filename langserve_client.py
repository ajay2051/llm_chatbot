from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable

chain_endpoint = "http://127.0.0.1:8000/langchain/"
chain = RemoteRunnable(chain_endpoint)
(chain.invoke("Tell me about Kathmandu"))

for chunk in chain.stream("What do you know about Nepal"):
    print(chunk)


prompt = PromptTemplate.from_template("Tell me about {topic}")

def format_prompt(inputs):
    prompt_value = prompt.format_prompt(**inputs)
    return prompt_value.to_string()

new_chain = RunnablePassthrough() | format_prompt | chain
(new_chain.invoke({"topic":"Tell me about Kathmandu"}))
