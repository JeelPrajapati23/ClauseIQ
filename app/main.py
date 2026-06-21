from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from app.loader import process_pdf
from app.database import save_chunks_to_vector_db,query_vector_db, get_reranking_retriever
from fastapi import Query
from pydantic import BaseModel
from app.generator import generate_answer, check_faithfulness

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
        # Retrieve the top 5 most relevant chunks
        retrieved_docs = query_vector_db(query=question, k=5)
        
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
        retriever = get_reranking_retriever()
        retrieved_docs = retriever.invoke(request.question)
        
        # GUARDRAIL: If retrieved context is empty -> explicitly refuse to answer
        if not retrieved_docs:
            return {
                "question": request.question,
                "answer": "I cannot answer this based on the provided documents. No relevant context was found.",
                "is_faithful": True, # Technically true as it refused safely
                "sources": []
            }
            
        context_parts = []
        sources_metadata = []
        
        for doc in retrieved_docs:
            source_file = doc.metadata.get("source", "Unknown").split("\\")[-1] 
            page = doc.metadata.get("page", "Unknown")
            context_parts.append(f"Source: {source_file}, Page: {page}\nContent: {doc.page_content}\n")
            sources_metadata.append({
                "file": source_file,
                "page": page,
                "content_preview": doc.page_content[:100] + "..."
            })
            
        formatted_context = "\n---\n".join(context_parts)
        
        # Generate the initial answer
        answer = generate_answer(request.question, formatted_context)
        
       # GUARDRAIL: Clear sources if the model refused to answer
        if "therefore an answer cannot be generated" in answer:
            sources_metadata = [] # Wipe the sources
            is_faithful = True    # It correctly refused, so it is faithful
        else:
            # Only run the auditor if it actually attempted an answer
            is_faithful = check_faithfulness(
                question=request.question, 
                context=formatted_context, 
                generated_answer=answer
            )
        
        return {
            "question": request.question,
            "answer": answer,
            "is_faithful_to_context": is_faithful, 
            "retrieved_sources": sources_metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))