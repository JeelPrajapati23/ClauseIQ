from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from app.loader import process_pdf
from app.database import save_chunks_to_vector_db
from app.database import save_chunks_to_vector_db, query_vector_db
from fastapi import Query
from pydantic import BaseModel
from app.generator import generate_answer

app = FastAPI(title="Ask My Docs RAG")

UPLOAD_DIR = "app/test_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-and-index/")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Extract and chunk the PDF
        chunks = process_pdf(file_path)
        
        # 2. Embed and save to Qdrant Vector Database
        save_chunks_to_vector_db(chunks)
        
        return {
            "filename": file.filename,
            "status": "Successfully chunked and embedded",
            "total_chunks_saved": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/query/")
async def query_documents(question: str = Query(..., description="The question to ask your documents")):
    try:
        # Retrieve the top 4 most relevant chunks
        retrieved_docs = query_vector_db(query=question, k=4)
        
        if not retrieved_docs:
            return {"status": "No relevant documents found."}
            
        # Format the results cleanly to return to the user
        formatted_results = []
        for i, doc in enumerate(retrieved_docs):
            formatted_results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "source_file": doc.metadata.get("source", "Unknown file"),
                "page_number": doc.metadata.get("page", "Unknown page")
            })
            
        return {
            "question": question,
            "retrieved_context": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class AskRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(request: AskRequest):
    try:
        # 1. Retrieve the top 4 chunks from Qdrant
        retrieved_docs = query_vector_db(query=request.question, k=4)
        
        if not retrieved_docs:
            return {
                "question": request.question,
                "answer": "No relevant documents found in the database to answer this.",
                "sources": []
            }
            
        # 2. Format the retrieved context for the LLM prompt
        context_parts = []
        sources_metadata = []
        
        for doc in retrieved_docs:
            # Extract just the filename from the full path for cleaner citations
            source_file = doc.metadata.get("source", "Unknown").split("\\")[-1] 
            page = doc.metadata.get("page", "Unknown")
            
            # This is the string injected into the {context} variable
            context_parts.append(f"Source: {source_file}, Page: {page}\nContent: {doc.page_content}\n")
            
            # This is the structured data returned to the frontend UI
            sources_metadata.append({
                "file": source_file,
                "page": page,
                "content_preview": doc.page_content[:100] + "..."
            })
            
        formatted_context = "\n---\n".join(context_parts)
        
        # 3. Pass everything to Groq to generate the final answer
        answer = generate_answer(request.question, formatted_context)
        
        return {
            "question": request.question,
            "answer": answer,
            "retrieved_sources": sources_metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))