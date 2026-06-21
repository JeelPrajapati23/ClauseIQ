import os
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Initialize the BGE embedding model.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},        # swap to "cuda" if a GPU is available
    encode_kwargs={"normalize_embeddings": True},
)


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
def get_reranking_retriever(collection_name="pdf_knowledge_base", initial_k=10, final_k=3):
    """Combines Vector, BM25, and Cohere Cross-Encoder Reranking."""
    
    # 1. Setup the Vector Retriever (Pull top 10)
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        check_compatibility=False
    )
    vector_retriever = qdrant.as_retriever(search_kwargs={"k": initial_k})
    
    # 2. Setup the BM25 Retriever (Pull top 10)
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL)
    records, _ = client.scroll(collection_name=collection_name, limit=2000) 
    
    docs = []
    for r in records:
        docs.append(Document(
            page_content=r.payload.get("page_content", ""), 
            metadata=r.payload.get("metadata", {})
        ))
        
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = initial_k
    
    # 3. Combine them using Reciprocal Rank Fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5] 
    )

    # 4. Initialize the Cohere Reranker
    compressor = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"), 
        model="rerank-english-v3.0", # Cohere's latest and most accurate model
        top_n=final_k # Compress the 10 chunks down to the 3 absolute best
    )
    
    # 5. Wrap the hybrid pipeline in the reranker
    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return reranking_retriever