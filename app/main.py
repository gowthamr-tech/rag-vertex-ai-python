from fastapi import FastAPI
from app.rag import build_qa_chain

app = FastAPI(title="RAG Vertex AI API")

vector_db, llm = build_qa_chain()


@app.get("/")
def health():
    return {"status": "RAG app is running"}


@app.post("/ask")
def ask_question(question: str):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)
    return {"answer": answer.content}
