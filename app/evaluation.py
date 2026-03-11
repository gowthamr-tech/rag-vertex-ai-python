import math
import re
from typing import Any

from pydantic import BaseModel, Field

from app.rag import get_answer, get_retrieved_documents


class EvaluationCase(BaseModel):
    question: str
    expected_answer: str | None = None
    expected_context_substrings: list[str] = Field(default_factory=list)
    retrieval_k: int = 3
    rerank_k: int | None = None
    use_reranking: bool = True


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase]


def normalize_text(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(value: str) -> list[str]:
    normalized = normalize_text(value)
    return normalized.split() if normalized else []


def build_token_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def count_overlap(predicted_tokens: list[str], expected_tokens: list[str]) -> int:
    expected_counts = build_token_counts(expected_tokens)
    overlap = 0
    for token in predicted_tokens:
        if expected_counts.get(token, 0) > 0:
            overlap += 1
            expected_counts[token] -= 1
    return overlap


def exact_match_score(predicted: str, expected: str) -> float:
    return float(normalize_text(predicted) == normalize_text(expected))


def answer_token_metrics(predicted: str, expected: str) -> dict[str, float]:
    predicted_tokens = tokenize(predicted)
    expected_tokens = tokenize(expected)

    if not predicted_tokens and not expected_tokens:
        return {
            "exact_match": 1.0,
            "token_precision": 1.0,
            "token_recall": 1.0,
            "token_f1": 1.0,
            "jaccard": 1.0,
        }

    overlap = count_overlap(predicted_tokens, expected_tokens)
    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(expected_tokens) if expected_tokens else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    predicted_set = set(predicted_tokens)
    expected_set = set(expected_tokens)
    union = predicted_set | expected_set
    jaccard = len(predicted_set & expected_set) / len(union) if union else 1.0

    return {
        "exact_match": exact_match_score(predicted, expected),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "jaccard": jaccard,
    }


def expected_match_flags(contexts: list[str], expected_context_substrings: list[str]) -> list[int]:
    normalized_contexts = [normalize_text(context) for context in contexts]
    flags: list[int] = []
    for context in normalized_contexts:
        is_relevant = any(
            normalize_text(expected) in context
            for expected in expected_context_substrings
        )
        flags.append(1 if is_relevant else 0)
    return flags


def retrieval_hit_at_k(flags: list[int]) -> float | None:
    if not flags:
        return None
    return float(any(flags))


def retrieval_precision_at_k(flags: list[int]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


def retrieval_recall_at_k(flags: list[int], expected_total: int) -> float | None:
    if expected_total <= 0:
        return None
    return min(sum(flags), expected_total) / expected_total


def retrieval_f1_at_k(precision_at_k: float | None, recall_at_k: float | None) -> float | None:
    if precision_at_k is None or recall_at_k is None:
        return None
    if precision_at_k + recall_at_k == 0:
        return 0.0
    return 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)


def reciprocal_rank(flags: list[int]) -> float | None:
    if not flags:
        return None
    for index, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / index
    return 0.0


def average_precision(flags: list[int], expected_total: int) -> float | None:
    if not flags or expected_total <= 0:
        return None

    hits = 0
    precision_sum = 0.0
    for index, flag in enumerate(flags, start=1):
        if flag:
            hits += 1
            precision_sum += hits / index

    if hits == 0:
        return 0.0
    return precision_sum / expected_total


def ndcg_at_k(flags: list[int], expected_total: int) -> float | None:
    if not flags or expected_total <= 0:
        return None

    dcg = sum(flag / math.log2(index + 1) for index, flag in enumerate(flags, start=1))
    ideal_flags = [1] * min(expected_total, len(flags))
    idcg = sum(flag / math.log2(index + 1) for index, flag in enumerate(ideal_flags, start=1))
    if idcg == 0:
        return None
    return dcg / idcg


def context_coverage(contexts: list[str], expected_context_substrings: list[str]) -> float | None:
    if not expected_context_substrings:
        return None

    normalized_context = " ".join(normalize_text(context) for context in contexts)
    matched = 0
    for expected in expected_context_substrings:
        if normalize_text(expected) in normalized_context:
            matched += 1
    return matched / len(expected_context_substrings)


def retrieval_metrics(contexts: list[str], expected_context_substrings: list[str]) -> dict[str, float | None]:
    if not expected_context_substrings:
        return {
            "retrieval_hit_at_k": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "retrieval_f1_at_k": None,
            "mrr": None,
            "map": None,
            "ndcg_at_k": None,
            "context_coverage": None,
        }

    flags = expected_match_flags(contexts, expected_context_substrings)
    expected_total = len(expected_context_substrings)
    precision_at_k = retrieval_precision_at_k(flags)
    recall_at_k = retrieval_recall_at_k(flags, expected_total)

    return {
        "retrieval_hit_at_k": retrieval_hit_at_k(flags),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "retrieval_f1_at_k": retrieval_f1_at_k(precision_at_k, recall_at_k),
        "mrr": reciprocal_rank(flags),
        "map": average_precision(flags, expected_total),
        "ndcg_at_k": ndcg_at_k(flags, expected_total),
        "context_coverage": context_coverage(contexts, expected_context_substrings),
    }


def evaluate_case(case: EvaluationCase) -> dict[str, Any]:
    retrieved_docs = get_retrieved_documents(
        question=case.question,
        k=case.retrieval_k,
        use_reranking=case.use_reranking,
        rerank_k=case.rerank_k,
    )
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
    generated_answer = get_answer(case.question, docs=retrieved_docs)

    metrics: dict[str, float | None] = retrieval_metrics(
        retrieved_contexts,
        case.expected_context_substrings,
    )

    if case.expected_answer:
        metrics.update(answer_token_metrics(generated_answer, case.expected_answer))

    return {
        "question": case.question,
        "generated_answer": generated_answer,
        "retrieved_contexts": retrieved_contexts,
        "metrics": metrics,
        "config": {
            "retrieval_k": case.retrieval_k,
            "rerank_k": case.rerank_k,
            "use_reranking": case.use_reranking,
        },
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for result in results:
        for metric_name, metric_value in result["metrics"].items():
            if metric_value is None:
                continue
            totals[metric_name] = totals.get(metric_name, 0.0) + metric_value
            counts[metric_name] = counts.get(metric_name, 0) + 1

    return {
        metric_name: totals[metric_name] / counts[metric_name]
        for metric_name in totals
        if counts[metric_name] > 0
    }


def evaluate_dataset(request: EvaluationRequest) -> dict[str, Any]:
    results = [evaluate_case(case) for case in request.cases]
    return {
        "summary": summarize_results(results),
        "results": results,
    }
