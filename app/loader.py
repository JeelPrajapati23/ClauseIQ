import os
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path: str):
    """Loads a PDF and splits it into manageable chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # 1. Load the PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Split the text with your specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )
    
    # split_documents applies the chunking logic directly to the loaded LangChain Document objects
    chunks = text_splitter.split_documents(documents)
    
    return chunks