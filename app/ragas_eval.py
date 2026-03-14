from typing import Any

from datasets import Dataset
from pydantic import BaseModel, Field
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.rag import get_answer, get_embeddings, get_llm, get_retrieved_documents


class RagasCase(BaseModel):
    question: str
    ground_truth: str
    expected_contexts: list[str] = Field(default_factory=list)
    retrieval_k: int = 3
    rerank_k: int | None = None
    use_reranking: bool = True


class RagasEvaluationRequest(BaseModel):
    cases: list[RagasCase]


def build_ragas_rows(request: RagasEvaluationRequest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in request.cases:
        retrieved_docs = get_retrieved_documents(
            question=case.question,
            k=case.retrieval_k,
            use_reranking=case.use_reranking,
            rerank_k=case.rerank_k,
        )
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        answer = get_answer(case.question, docs=retrieved_docs)

        # Ragas expects all context fields as lists of strings.
        reference_contexts = case.expected_contexts or retrieved_contexts

        rows.append(
            {
                "question": case.question,
                "answer": answer,
                "ground_truth": case.ground_truth,
                "contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
                "retrieved_contexts": retrieved_contexts,
            }
        )
    return rows


def evaluate_with_ragas(request: RagasEvaluationRequest) -> dict[str, Any]:
    rows = build_ragas_rows(request)
    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=get_llm(),
        embeddings=get_embeddings(),
    )

    summary = result.to_pandas().mean(numeric_only=True).to_dict()
    return {
        "summary": summary,
        "results": result.to_pandas().to_dict(orient="records"),
        "rows": rows,
    }
