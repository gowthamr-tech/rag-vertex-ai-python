from fastapi import FastAPI
import uvicorn
import os
from contextlib import asynccontextmanager
from app.rag import build_qa_chain

# This dictionary will hold our RAG components
rag_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the RAG chain when the app starts
    print("Starting RAG chain initialization...")
    vector_db, llm = build_qa_chain()
    rag_components["vector_db"] = vector_db
    rag_components["llm"] = llm
    print("RAG chain ready!")
    yield
    # Clean up on shutdown if needed
    rag_components.clear()

app = FastAPI(title="RAG Vertex AI API", lifespan=lifespan)

@app.get("/")
def health():
    return {"status": "RAG app is running"}

@app.post("/ask")
def ask_question(question: str):
    vector_db = rag_components.get("vector_db")
    llm = rag_components.get("llm")
    
    if not vector_db or not llm:
        return {"error": "RAG chain is still initializing. Please try again in a moment."}

    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    answer = llm.invoke(prompt)
    return {"answer": answer.content}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)