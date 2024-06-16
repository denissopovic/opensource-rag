from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3", request_timeout=120.0)
response = llm.complete("Who is Denis Ragovic?")
print(response)