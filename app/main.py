import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.evaluation import EvaluationRequest, evaluate_dataset
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
async def ask_question(
    question: str,
    use_reranking: bool = False,
    rerank_k: int | None = None,
):
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        answer = get_answer(
            question,
            use_reranking=use_reranking,
            rerank_k=rerank_k,
        )
        print(f"Question: {question} | Answer: {answer}")
        return {
            "question": question,
            "answer": answer,
            "use_reranking": use_reranking,
            "rerank_k": rerank_k,
        }
    except Exception as e:
        print(f"Ask Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate(request: EvaluationRequest):
    if not request.cases:
        raise HTTPException(status_code=400, detail="At least one evaluation case is required.")

    try:
        return evaluate_dataset(request)
    except Exception as e:
        print(f"Evaluation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
