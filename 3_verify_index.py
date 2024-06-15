import qdrant_client
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize Qdrant client
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="cv")

# Initialize LLM
llm = Ollama(model="llama3")

# Provide a model name for the HuggingFaceEmbedding
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create service context with the embedding model
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

# Create the index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

# Query the index
query_engine = index.as_query_engine(similarity_top_k=20)
response = query_engine.query("Does anybody like Drupal? Give details.")
print(response)
