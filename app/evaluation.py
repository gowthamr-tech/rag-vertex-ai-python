import math
import re
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.rag import get_answer, get_embeddings, get_retrieved_documents


class EvaluationCase(BaseModel):
    question: str
    expected_answer: str | None = None

    # Legacy evaluation input. Kept for backward compatibility and simple demos.
    expected_context_substrings: list[str] = Field(default_factory=list)

    # Production-style evaluation inputs. Prefer these when creating gold datasets.
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_titles: list[str] = Field(default_factory=list)

    retrieval_k: int = 3
    rerank_k: int | None = None
    use_reranking: bool = True


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase]


_EMBEDDINGS_CLIENT = None
_EMBEDDING_CACHE: dict[str, list[float]] = {}


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


def get_embeddings_client():
    global _EMBEDDINGS_CLIENT
    if _EMBEDDINGS_CLIENT is None:
        _EMBEDDINGS_CLIENT = get_embeddings()
    return _EMBEDDINGS_CLIENT


def embed_text(value: str) -> list[float]:
    normalized_value = value.strip()
    if not normalized_value:
        return []

    cached = _EMBEDDING_CACHE.get(normalized_value)
    if cached is not None:
        return cached

    embedding = get_embeddings_client().embed_query(normalized_value)
    _EMBEDDING_CACHE[normalized_value] = embedding
    return embedding


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot_product / (left_norm * right_norm)


def semantic_similarity_score(left: str, right: str) -> float | None:
    if not left.strip() or not right.strip():
        return None
    return cosine_similarity(embed_text(left), embed_text(right))


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


def answer_semantic_metrics(predicted: str, expected: str) -> dict[str, float | None]:
    return {
        "semantic_similarity": semantic_similarity_score(predicted, expected),
    }


def normalize_references(values: list[str]) -> set[str]:
    return {normalize_text(value) for value in values if normalize_text(value)}


def get_document_references(doc: Document) -> dict[str, str]:
    metadata = doc.metadata or {}
    references = {
        "document_id": normalize_text(str(getattr(doc, "id", "") or "")),
        "source": normalize_text(str(metadata.get("source", "") or "")),
        "title": normalize_text(str(metadata.get("title", "") or "")),
    }
    return references


def get_expected_reference_groups(case: EvaluationCase) -> dict[str, set[str]]:
    return {
        "document_id": normalize_references(case.expected_document_ids),
        "source": normalize_references(case.expected_sources),
        "title": normalize_references(case.expected_titles),
    }


def count_expected_targets(reference_groups: dict[str, set[str]]) -> int:
    return sum(len(values) for values in reference_groups.values())


def match_document_targets(doc: Document, case: EvaluationCase) -> set[str]:
    references = get_document_references(doc)
    expected_groups = get_expected_reference_groups(case)
    matched_targets: set[str] = set()

    for ref_type, expected_values in expected_groups.items():
        doc_value = references[ref_type]
        if doc_value and doc_value in expected_values:
            matched_targets.add(f"{ref_type}:{doc_value}")

    return matched_targets


# Legacy evaluator retained for backward compatibility.
# This is useful for quick experiments, but production evaluation should prefer
# expected_document_ids / expected_sources / expected_titles instead of substring matching.
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


def legacy_matched_targets(context: str, expected_context_substrings: list[str]) -> set[str]:
    normalized_context = normalize_text(context)
    matched_targets: set[str] = set()
    for expected in expected_context_substrings:
        normalized_expected = normalize_text(expected)
        if normalized_expected and normalized_expected in normalized_context:
            matched_targets.add(f"context:{normalized_expected}")
    return matched_targets


def build_relevance_trace(docs: list[Document], case: EvaluationCase) -> tuple[list[int], list[set[str]], str]:
    reference_groups = get_expected_reference_groups(case)
    print(f"Reference Groups: {reference_groups}")
    expected_target_count = count_expected_targets(reference_groups)
    print(f"Expected Target Count: {expected_target_count}")

    matched_targets_per_rank: list[set[str]] = []
    if expected_target_count > 0:
        evaluation_mode = "metadata"
        for doc in docs:
            matched_targets_per_rank.append(match_document_targets(doc, case))
    else:
        evaluation_mode = "legacy_substring"
        for doc in docs:
            matched_targets_per_rank.append(
                legacy_matched_targets(doc.page_content, case.expected_context_substrings)
            )

    seen_targets: set[str] = set()
    novelty_flags: list[int] = []
    print(f"Matched Targets Per Rank: {matched_targets_per_rank}")
    for matched_targets in matched_targets_per_rank:
        unseen_targets = matched_targets - seen_targets
        novelty_flags.append(1 if unseen_targets else 0)
        seen_targets.update(unseen_targets)

    return novelty_flags, matched_targets_per_rank, evaluation_mode


def retrieval_hit_at_k(flags: list[int]) -> float | None:
    if not flags:
        return None
    return float(any(flags))


def retrieval_precision_at_k(flags: list[int]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


def retrieval_recall_at_k(unique_matches: int, expected_total: int) -> float | None:
    if expected_total <= 0:
        return None
    return unique_matches / expected_total


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
    ideal_hits = min(expected_total, len(flags))
    ideal_flags = [1] * ideal_hits + [0] * (len(flags) - ideal_hits)
    idcg = sum(flag / math.log2(index + 1) for index, flag in enumerate(ideal_flags, start=1))
    if idcg == 0:
        return None
    return dcg / idcg


def context_coverage(unique_matches: int, expected_total: int) -> float | None:
    if expected_total <= 0:
        return None
    return unique_matches / expected_total


def retrieval_metrics(docs: list[Document], case: EvaluationCase) -> tuple[dict[str, float | None], dict[str, Any]]:
    novelty_flags, matched_targets_per_rank, evaluation_mode = build_relevance_trace(docs, case)
    unique_matches = len(set().union(*matched_targets_per_rank)) if matched_targets_per_rank else 0

    expected_total = count_expected_targets(get_expected_reference_groups(case))
    if expected_total == 0 and case.expected_context_substrings:
        expected_total = len(normalize_references(case.expected_context_substrings))

    if expected_total == 0:
        return (
            {
                "retrieval_hit_at_k": None,
                "precision_at_k": None,
                "recall_at_k": None,
                "retrieval_f1_at_k": None,
                "mrr": None,
                "map": None,
                "ndcg_at_k": None,
                "context_coverage": None,
            },
            {
                "evaluation_mode": evaluation_mode,
                "matched_targets_per_rank": [],
            },
        )

    precision_at_k = retrieval_precision_at_k(novelty_flags)
    recall_at_k = retrieval_recall_at_k(unique_matches, expected_total)

    metrics = {
        "retrieval_hit_at_k": retrieval_hit_at_k(novelty_flags),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "retrieval_f1_at_k": retrieval_f1_at_k(precision_at_k, recall_at_k),
        "mrr": reciprocal_rank(novelty_flags),
        "map": average_precision(novelty_flags, expected_total),
        "ndcg_at_k": ndcg_at_k(novelty_flags, expected_total),
        "context_coverage": context_coverage(unique_matches, expected_total),
    }
    if case.expected_context_substrings:
        metrics["semantic_context_similarity"] = semantic_similarity_score(
            " ".join(doc.page_content for doc in docs),
            " ".join(case.expected_context_substrings),
        )
    diagnostics = {
        "evaluation_mode": evaluation_mode,
        "matched_targets_per_rank": [sorted(matches) for matches in matched_targets_per_rank],
    }
    return metrics, diagnostics


def serialize_retrieved_docs(docs: list[Document]) -> list[dict[str, Any]]:
    serialized_docs: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs, start=1):
        serialized_docs.append(
            {
                "rank": rank,
                "document_id": getattr(doc, "id", None),
                "metadata": doc.metadata,
                "content": doc.page_content,
            }
        )
    return serialized_docs


def evaluate_case(case: EvaluationCase) -> dict[str, Any]:
    retrieved_docs = get_retrieved_documents(
        question=case.question, 
        k=case.retrieval_k,
        use_reranking=case.use_reranking,
        rerank_k=case.rerank_k,
    )
    generated_answer = get_answer(case.question, docs=retrieved_docs)

    metrics, retrieval_diagnostics = retrieval_metrics(retrieved_docs, case)
    if case.expected_answer:
        metrics.update(answer_token_metrics(generated_answer, case.expected_answer))
        metrics.update(answer_semantic_metrics(generated_answer, case.expected_answer))

    return {
        "question": case.question,
        "generated_answer": generated_answer,
        "retrieved_docs": serialize_retrieved_docs(retrieved_docs),
        "metrics": metrics,
        "retrieval_diagnostics": retrieval_diagnostics,
        "config": {
            "retrieval_k": case.retrieval_k,
            "rerank_k": case.rerank_k,
            "use_reranking": case.use_reranking,
        },
        "expected_references": {
            "document_ids": case.expected_document_ids,
            "sources": case.expected_sources,
            "titles": case.expected_titles,
            "context_substrings": case.expected_context_substrings,
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
