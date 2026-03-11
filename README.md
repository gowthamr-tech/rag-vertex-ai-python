# RAG Vertex AI Service

A FastAPI-based RAG service that:

- uploads and indexes PDF documents in Pinecone
- retrieves relevant chunks with Vertex AI embeddings
- generates answers with Vertex AI
- evaluates retrieval and answer quality
- supports optional reranking before answer generation

## Project Structure

```text
app/
  main.py         FastAPI endpoints
  rag.py          ingestion, retrieval, reranking, answer generation
  evaluation.py   evaluation request models and metrics
requirements.txt  Python dependencies
Dockerfile        container setup
```

## Requirements

- Python 3.9+
- Pinecone index and API key
- Google Cloud project with Vertex AI enabled

## Environment Variables

Create a `.env` file in the project root:

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Service URL:

```text
http://127.0.0.1:8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```bash
curl http://127.0.0.1:8000/
```

### Upload PDF

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/absolute/path/to/document.pdf"
```

This:

- loads the PDF
- splits it into chunks
- embeds the chunks with Vertex AI
- stores the vectors in Pinecone

### Ask a Question

Without reranking:

```bash
curl -X POST "http://127.0.0.1:8000/ask?question=Does%20Chennai%20Metro%20support%20PhonePe%20payments%3F&use_reranking=false"
```

With reranking:

```bash
curl -X POST "http://127.0.0.1:8000/ask?question=Does%20Chennai%20Metro%20support%20PhonePe%20payments%3F&use_reranking=true&rerank_k=10"
```

How it works:

1. retrieve chunks from Pinecone
2. optionally rerank the retrieved candidates
3. build a context string from the final chunks
4. send the context and question to Vertex AI
5. return the generated answer

## Evaluation

The `/evaluate` endpoint runs a dataset of test cases and returns:

- `summary`: average metric values across all cases
- `results`: per-case answer, retrieved contexts, and metrics

### Example Evaluation Request

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        "question": "Does Chennai Metro support PhonePe payments?",
        "expected_answer": "Yes, Chennai Metro supports PhonePe payments.",
        "expected_context_substrings": ["Phonepe"],
        "retrieval_k": 3,
        "rerank_k": 10,
        "use_reranking": true
      },
      {
        "question": "What support sections are available on the BookMyShow page?",
        "expected_answer": "The page includes FAQs, Terms and Conditions, and Privacy Policy.",
        "expected_context_substrings": [
          "FAQs",
          "Terms and Conditions",
          "Privacy Policy"
        ],
        "retrieval_k": 3,
        "rerank_k": 10,
        "use_reranking": true
      }
    ]
  }'
```

### Compare Before vs After Reranking

Run the same evaluation dataset twice:

- once with `use_reranking: false`
- once with `use_reranking: true`

Then compare the `summary` values.

Useful comparison metrics:

- retrieval: `retrieval_hit_at_k`, `precision_at_k`, `recall_at_k`, `mrr`
- answer: `exact_match`, `token_precision`, `token_recall`, `token_f1`, `jaccard`

## Implemented Metrics

### Retrieval Metrics

- `retrieval_hit_at_k`
- `precision_at_k`
- `recall_at_k`
- `retrieval_f1_at_k`
- `mrr`
- `map`
- `ndcg_at_k`
- `context_coverage`

### Answer Metrics

- `exact_match`
- `token_precision`
- `token_recall`
- `token_f1`
- `jaccard`

## Reranking

The current reranker is a lightweight lexical reranker implemented in [`app/rag.py`](/Users/Apple/Documents/rag_vertex_project/app/rag.py).

It:

- fetches more candidate chunks from Pinecone
- scores them using query-term overlap and term density
- returns the best final top `k` chunks

Why it helps:

- vector search may return semantically similar but noisy chunks
- reranking improves ordering of retrieved chunks
- better context often improves answer quality

## Current Limitations

- retrieval quality depends heavily on the quality of indexed content
- noisy footer/navigation text can dominate search results
- reranking can only reorder retrieved candidates; it cannot recover missing relevant chunks
- evaluation quality depends on realistic `expected_answer` and `expected_context_substrings`

## Suggested Next Improvements

- store metadata like source file, page number, and chunk id
- deduplicate repeated chunks before indexing
- filter boilerplate web content
- add a script to compare baseline vs reranked evaluation automatically
- add a stronger reranker such as a cross-encoder or LLM-based reranker
