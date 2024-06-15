from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import qdrant_client
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Re-initialize the vector store
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="cv")

# Initialize the LLM
llm = Ollama(model="llama3")

# Provide a model name for the HuggingFaceEmbedding
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create service context with the embedding model
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

# Load the index from the vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# This is just so you can easily tell the app is running
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/process_form', methods=['POST'])
@cross_origin()
def process_form():
    query = request.form.get('query')
    if query is not None:
        query_engine = index.as_query_engine(similarity_top_k=20)
        response = query_engine.query(query)
        return jsonify({"response": str(response)})
    else:
        return jsonify({"error": "query field is missing"}), 400

if __name__ == '__main__':
    app.run()
