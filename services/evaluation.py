"""
Assistente Jurídico SaaS IA V3.1
services/evaluation.py

Sistema de avaliação das respostas do RAG.

Métricas:

    1. Context Relevance
    2. Citation Coverage
    3. Groundedness
    4. Overall Score

Características:
- Heurísticas determinísticas.
- Não depende de LLM.
- Não depende de Streamlit.
- Compatível com rag_pipeline.py.
- Compatível com citações [1], [2], [3].
- Normalização segura dos chunks.
- Tratamento de entradas inválidas.
- Scores sempre entre 0 e 1.
- Resultado estruturado para Dashboard.
- Preparado para futura integração RAGAS/LLM-as-Judge.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TOKEN_PATTERN = r"[a-zA-ZÀ-ÿ0-9]+"

DEFAULT_WEIGHTS = {
    "context_relevance": 0.40,
    "citation_coverage": 0.30,
    "groundedness": 0.30,
}


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def _normalize_text(
    text: Any,
) -> str:
    """
    Normaliza um texto para avaliação lexical.
    """

    if text is None:
        return ""

    try:
        text = str(text)
    except Exception:
        return ""

    return text.strip().lower()


def _tokens(
    text: Any,
) -> set[str]:
    """
    Extrai tokens normalizados.
    """

    text = _normalize_text(
        text
    )

    if not text:
        return set()

    return set(
        re.findall(
            TOKEN_PATTERN,
            text,
        )
    )


def _chunk_content(
    chunk: Any,
) -> str:
    """
    Obtém o conteúdo de um chunk em diferentes formatos.

    Compatível com:
        dict
        page_content
        content
        text
    """

    if chunk is None:
        return ""

    if isinstance(
        chunk,
        dict,
    ):

        content = (
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("page_content")
            or chunk.get("chunk")
            or ""
        )

        return _normalize_text(
            content
        )

    if hasattr(
        chunk,
        "page_content",
    ):

        try:
            return _normalize_text(
                getattr(
                    chunk,
                    "page_content",
                )
            )
        except Exception:
            return ""

    if hasattr(
        chunk,
        "content",
    ):

        try:
            return _normalize_text(
                getattr(
                    chunk,
                    "content",
                )
            )
        except Exception:
            return ""

    try:
        return _normalize_text(
            chunk
        )
    except Exception:
        return ""


def _normalize_chunks(
    chunks: Sequence[Any] | None,
) -> List[Any]:
    """
    Remove chunks inválidos ou vazios.
    """

    if not chunks:
        return []

    valid = []

    for chunk in chunks:

        if _chunk_content(
            chunk
        ):

            valid.append(
                chunk
            )

    return valid


# ============================================================
# CONTEXT RELEVANCE
# ============================================================

def context_relevance(
    question: str,
    chunks: Sequence[Any] | None,
) -> float:
    """
    Mede quanto o conteúdo recuperado possui relação lexical
    com a pergunta.

    Resultado:
        0.0 = nenhuma relação
        1.0 = alta sobreposição lexical

    Observação:
        Esta é uma métrica heurística e não substitui
        avaliação semântica por LLM.
    """

    question_tokens = _tokens(
        question
    )

    normalized_chunks = (
        _normalize_chunks(
            chunks
        )
    )

    if (
        not question_tokens
        or not normalized_chunks
    ):
        return 0.0

    values = []

    for chunk in normalized_chunks:

        content_tokens = _tokens(
            _chunk_content(
                chunk
            )
        )

        if not content_tokens:

            values.append(
                0.0
            )

            continue

        intersection = (
            question_tokens
            & content_tokens
        )

        score = (
            len(intersection)
            / max(
                1,
                len(question_tokens),
            )
        )

        values.append(
            min(
                1.0,
                float(score),
            )
        )

    if not values:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            sum(values)
            / len(values),
        ),
    )


# ============================================================
# CITATION COVERAGE
# ============================================================

def _extract_citation_marks(
    answer: str,
) -> set[str]:
    """
    Extrai referências no formato:

        [1]
        [2]
        [15]
    """

    if not answer:
        return set()

    return set(
        re.findall(
            r"\[(\d+)\]",
            str(answer),
        )
    )


def _citation_id(
    citation: Any,
) -> str | None:
    """
    Extrai o ID de uma citação.
    """

    if citation is None:
        return None

    if isinstance(
        citation,
        dict,
    ):

        value = citation.get(
            "id"
        )

    else:

        value = getattr(
            citation,
            "id",
            None,
        )

    if value is None:
        return None

    return str(
        value
    )


def citation_coverage(
    answer: str,
    citations: Sequence[Any] | None,
) -> float:
    """
    Mede a proporção das citações utilizadas na resposta
    que realmente existem no conjunto de citações disponíveis.

    Exemplo:

        Resposta:
            "... conforme [1] e [2]."

        Citações:
            [1], [2], [3]

        Resultado:
            1.0
    """

    if not answer:
        return 0.0

    marks = _extract_citation_marks(
        answer
    )

    if not marks:
        return 0.0

    valid = set()

    for citation in citations or []:

        citation_id = _citation_id(
            citation
        )

        if citation_id is not None:

            valid.add(
                citation_id
            )

    if not valid:
        return 0.0

    coverage = (
        len(
            marks & valid
        )
        / len(marks)
    )

    return max(
        0.0,
        min(
            1.0,
            float(coverage),
        ),
    )


# ============================================================
# GROUNDEDNESS
# ============================================================

def groundedness(
    answer: str,
    chunks: Sequence[Any] | None,
) -> float:
    """
    Mede quanto do vocabulário da resposta aparece
    no contexto recuperado.

    É uma heurística simples.

    Não determina juridicamente se a resposta está correta.
    """

    answer_tokens = _tokens(
        answer
    )

    normalized_chunks = (
        _normalize_chunks(
            chunks
        )
    )

    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()

    for chunk in normalized_chunks:

        context_tokens |= _tokens(
            _chunk_content(
                chunk
            )
        )

    if not context_tokens:
        return 0.0

    overlap = (
        answer_tokens
        & context_tokens
    )

    score = (
        len(overlap)
        / len(answer_tokens)
    )

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


# ============================================================
# SCORE GERAL
# ============================================================

def overall_score(
    context_score: float,
    citation_score: float,
    grounded_score: float,
) -> float:
    """
    Calcula o score geral da avaliação.
    """

    score = (
        DEFAULT_WEIGHTS[
            "context_relevance"
        ]
        * context_score
        +
        DEFAULT_WEIGHTS[
            "citation_coverage"
        ]
        * citation_score
        +
        DEFAULT_WEIGHTS[
            "groundedness"
        ]
        * grounded_score
    )

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

def _score_label(
    score: float,
) -> str:
    """
    Classifica o resultado geral.
    """

    if score >= 0.85:
        return "excellent"

    if score >= 0.70:
        return "good"

    if score >= 0.50:
        return "moderate"

    return "low"


# ============================================================
# AVALIAÇÃO PRINCIPAL
# ============================================================

def evaluate_answer(
    question: str,
    answer: str,
    chunks: Sequence[Any] | None,
    citations: Sequence[Any] | None,
) -> Dict[str, Any]:
    """
    Avalia uma resposta gerada pelo RAG.

    Retorna:

        {
            "context_relevance": 0.0,
            "citation_coverage": 0.0,
            "groundedness": 0.0,
            "overall": 0.0,
            "label": "...",
            "method": "...",
        }
    """

    question = (
        question
        or ""
    ).strip()

    answer = (
        answer
        or ""
    ).strip()

    chunks = (
        chunks
        or []
    )

    citations = (
        citations
        or []
    )

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    context_score = context_relevance(
        question,
        chunks,
    )

    citation_score = citation_coverage(
        answer,
        citations,
    )

    grounded_score = groundedness(
        answer,
        chunks,
    )

    overall = overall_score(
        context_score,
        citation_score,
        grounded_score,
    )

    return {
        "context_relevance": round(
            context_score,
            3,
        ),

        "citation_coverage": round(
            citation_score,
            3,
        ),

        "groundedness": round(
            grounded_score,
            3,
        ),

        "overall": round(
            overall,
            3,
        ),

        "label": _score_label(
            overall
        ),

        "method": (
            "heuristic_baseline"
        ),

        "evaluator_version": (
            "V3.1"
        ),

        "weights": dict(
            DEFAULT_WEIGHTS
        ),

        "question_tokens": len(
            _tokens(
                question
            )
        ),

        "answer_tokens": len(
            _tokens(
                answer
            )
        ),

        "chunk_count": len(
            _normalize_chunks(
                chunks
            )
        ),

        "citation_count": len(
            citations
        ),
    }


# ============================================================
# AVALIAÇÃO DO RESULTADO RAG
# ============================================================

def evaluate_rag_result(
    result: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Avalia diretamente o resultado produzido pelo
    rag_pipeline.rag_answer().

    Isso facilita a integração:

        result = rag_answer(...)

        evaluation = evaluate_rag_result(result)
    """

    if not isinstance(
        result,
        dict,
    ):

        return {
            "context_relevance": 0.0,
            "citation_coverage": 0.0,
            "groundedness": 0.0,
            "overall": 0.0,
            "label": "low",
            "method": "heuristic_baseline",
            "evaluator_version": "V3.1",
            "error": "Resultado RAG inválido.",
        }

    evaluation = evaluate_answer(
        question=result.get(
            "query",
            "",
        ),
        answer=result.get(
            "answer",
            "",
        ),
        chunks=result.get(
            "context",
            [],
        ),
        citations=result.get(
            "citations",
            [],
        ),
    )

    evaluation[
        "rag_evidence_status"
    ] = result.get(
        "evidence_status"
    )

    evaluation[
        "answer_generated"
    ] = bool(
        result.get(
            "answer_generated",
            False,
        )
    )

    evaluation[
        "rag_error"
    ] = result.get(
        "error"
    )

    return evaluation


# ============================================================
# COMPARAÇÃO DE RESPOSTAS
# ============================================================

def compare_evaluations(
    current: Dict[str, Any],
    previous: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compara duas avaliações.

    Útil para:
        - testes de regressão;
        - avaliação de versões;
        - dashboard;
        - monitoramento.
    """

    metrics = [
        "context_relevance",
        "citation_coverage",
        "groundedness",
        "overall",
    ]

    result = {}

    for metric in metrics:

        current_value = float(
            current.get(
                metric,
                0.0,
            )
            or 0.0
        )

        previous_value = float(
            previous.get(
                metric,
                0.0,
            )
            or 0.0
        )

        result[
            metric
        ] = {
            "current": round(
                current_value,
                3,
            ),
            "previous": round(
                previous_value,
                3,
            ),
            "delta": round(
                current_value
                - previous_value,
                3,
            ),
        }

    return result


# ============================================================
# TESTE UNITÁRIO SIMPLES
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Executa teste estrutural e funcional básico.

    Não utiliza LLM, embeddings ou FAISS.
    """

    question = (
        "Qual é o prazo para contestação?"
    )

    answer = (
        "O prazo para contestação é de "
        "quinze dias úteis, conforme [1]."
    )

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document": "teste.pdf",
            "page": 1,
            "content": (
                "O prazo para contestação "
                "é de quinze dias úteis."
            ),
        }
    ]

    citations = [
        {
            "id": 1,
            "document": "teste.pdf",
            "page": 1,
        }
    ]

    result = evaluate_answer(
        question=question,
        answer=answer,
        chunks=chunks,
        citations=citations,
    )

    valid = (
        0.0
        <= result["context_relevance"]
        <= 1.0
        and
        0.0
        <= result["citation_coverage"]
        <= 1.0
        and
        0.0
        <= result["groundedness"]
        <= 1.0
        and
        0.0
        <= result["overall"]
        <= 1.0
    )

    return {
        "module": "services.evaluation",
        "version": "V3.1",
        "status": (
            "ok"
            if valid
            else "error"
        ),
        "valid_scores": valid,
        "result": result,
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print(
        "EVALUATION.PY V3.1 - SELF TEST"
    )
    print("=" * 60)

    print(
        f"Status              : "
        f"{result['status']}"
    )

    print(
        f"Scores válidos      : "
        f"{result['valid_scores']}"
    )

    evaluation = result[
        "result"
    ]

    print(
        f"Context relevance   : "
        f"{evaluation['context_relevance']}"
    )

    print(
        f"Citation coverage   : "
        f"{evaluation['citation_coverage']}"
    )

    print(
        f"Groundedness        : "
        f"{evaluation['groundedness']}"
    )

    print(
        f"Overall             : "
        f"{evaluation['overall']}"
    )

    print(
        f"Classificação       : "
        f"{evaluation['label']}"
    )

    print("=" * 60)
