from pathlib import Path
import qdrant_client
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
    SimpleDirectoryReader,
    StorageContext
)
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Download and load data
JSONReader = download_loader("JSONReader")
loader = JSONReader()
documents = SimpleDirectoryReader("./data/pdf").load_data()

# Initialize Qdrant client
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="cv")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize LLM
llm = Ollama(model="llama3")

# Provide a model name for the HuggingFaceEmbedding
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create service context with the embedding model
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

# Create the index
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("Who knows drupal? Give details.")
print(response)
