from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


QDRANT_URL = "http://localhost:6333"

def save_chunks_to_vector_db(chunks, collection_name="pdf_knowledge_base"):
    """Converts text chunks to embeddings and saves them to the Docker Qdrant instance."""
    
   
    qdrant = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
        force_recreate=True, #overwrite old test data
        check_compatibility=False
    )
    
    return qdrant

def query_vector_db(query: str, k: int = 4, collection_name: str = "pdf_knowledge_base"):
    """Connects to the existing Qdrant collection and retrieves the top-k similar chunks."""
    
    # Connect to the existing collection instead of creating a new one
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        check_compatibility=False
    )
    
    # Perform the similarity search
    # This automatically converts the text query into an embedding and compares it
    results = qdrant.similarity_search(query=query, k=k)
    
    return results