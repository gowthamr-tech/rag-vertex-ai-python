import csv
import json
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.evaluation import EvaluationRequest, evaluate_dataset
from app.ragas_eval import RagasEvaluationRequest, evaluate_with_ragas
from app.rag import (
    build_documents_from_records,
    clear_pinecone_index,
    ingest_documents,
    ingest_pdf,
    get_answer,
)

app = FastAPI(title="RAG Vertex AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DatasetIngestRequest(BaseModel):
    records: list[dict]
    text_field: str
    metadata_fields: list[str] = Field(default_factory=list)
    static_metadata: dict = Field(default_factory=dict)
    chunk_size: int = 800
    chunk_overlap: int = 80


def parse_uploaded_records(filename: str, raw_bytes: bytes) -> list[dict]:
    lowered_name = filename.lower()
    decoded = raw_bytes.decode("utf-8")

    if lowered_name.endswith(".json"):
        payload = json.loads(decoded)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload["records"]
        raise ValueError("JSON file must contain either a list of records or an object with a records list.")

    if lowered_name.endswith(".jsonl"):
        return [json.loads(line) for line in decoded.splitlines() if line.strip()]

    if lowered_name.endswith(".csv"):
        reader = csv.DictReader(decoded.splitlines())
        return [dict(row) for row in reader]

    raise ValueError("Supported dataset formats are .json, .jsonl, and .csv.")

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


@app.post("/ingest-records")
async def ingest_records(request: DatasetIngestRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="records cannot be empty.")

    try:
        documents = build_documents_from_records(
            records=request.records,
            text_field=request.text_field,
            metadata_fields=request.metadata_fields,
            static_metadata=request.static_metadata,
        )
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents were created from the supplied records.")

        chunk_count = ingest_documents(
            documents=documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return {
            "message": "Dataset records indexed successfully.",
            "records_received": len(request.records),
            "documents_indexed": len(documents),
            "chunks_created": chunk_count,
            "text_field": request.text_field,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ingest Records Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    text_field: str = "context",
    metadata_fields: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 80,
):
    try:
        raw_bytes = await file.read()
        records = parse_uploaded_records(file.filename or "", raw_bytes)
        selected_metadata_fields = [field.strip() for field in metadata_fields.split(",") if field.strip()]

        documents = build_documents_from_records(
            records=records,
            text_field=text_field,
            metadata_fields=selected_metadata_fields,
            static_metadata={"uploaded_filename": file.filename},
        )
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents were created from the uploaded dataset.")

        chunk_count = ingest_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return {
            "message": f"Successfully indexed dataset file {file.filename}",
            "records_received": len(records),
            "documents_indexed": len(documents),
            "chunks_created": chunk_count,
            "text_field": text_field,
            "metadata_fields": selected_metadata_fields,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Upload Dataset Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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


@app.post("/evaluate-ragas")
async def evaluate_ragas(request: RagasEvaluationRequest):
    if not request.cases:
        raise HTTPException(status_code=400, detail="At least one Ragas evaluation case is required.")

    try:
        return evaluate_with_ragas(request)
    except Exception as e:
        print(f"Ragas Evaluation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/clear-index")
async def clear_index(namespace: str | None = None):
    try:
        clear_pinecone_index(namespace=namespace)
        return {
            "message": "Pinecone vectors cleared successfully.",
            "index_name": os.getenv("PINECONE_INDEX_NAME"),
            "namespace": namespace,
        }
    except Exception as e:
        print(f"Clear Index Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
