import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.rag import ingest_pdf, get_answer

app = FastAPI(title="RAG Vertex AI Service")

@app.get("/")
def health_check():
    return {"status": "running"}

# 🔹 Endpoint to upload and index the PDF
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_path = f"/tmp/{file.filename}"
    
    try:
        # Save file to temporary storage
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest into Pinecone
        num_chunks = ingest_pdf(temp_path)
        
        return {
            "message": f"Successfully indexed {file.filename}",
            "chunks_created": num_chunks
        }
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the file from /tmp to save memory
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 🔹 Endpoint to ask questions
@app.post("/ask")
async def ask_question(question: str):
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        answer = get_answer(question)
        print(f"Question: {question} | Answer: {answer}")
        return {"question": question, "answer": answer}
    except Exception as e:
        print(f"Ask Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))