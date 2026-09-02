from functools import lru_cache
from typing import Any

import os
import re


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

RERANK_BATCH_SIZE = int(
    os.getenv("RERANK_BATCH_SIZE", "16")
)


# ============================================================
# MODELO
# ============================================================

@lru_cache(maxsize=1)
def _load_reranker():
    """
    Carrega o CrossEncoder uma única vez por processo.

    Antes:
        O modelo era carregado a cada consulta.

    Agora:
        O modelo permanece em memória e é reutilizado.
    """

    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normaliza texto para o fallback lexical.
    """

    text = text or ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokens(text: str) -> set[str]:
    """
    Extrai tokens simples para fallback lexical.
    """

    return set(
        re.findall(
            r"[a-zA-ZÀ-ÿ0-9]+",
            _normalize_text(text),
        )
    )


# ============================================================
# FALLBACK LEXICAL
# ============================================================

def _lexical_rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int,
):
    """
    Reranker lexical utilizado quando o CrossEncoder
    não está disponível.

    Não substitui semanticamente o CrossEncoder,
    mas mantém o sistema operacional.
    """

    query_tokens = _tokens(query)

    if not query_tokens:
        return documents[:top_k]

    results = []

    for document in documents:

        content = document.get(
            "content",
            "",
        )

        document_tokens = _tokens(content)

        intersection = query_tokens & document_tokens

        score = (
            len(intersection)
            / max(1, len(query_tokens))
        )

        item = dict(document)

        item["reranker_score"] = float(score)

        # Identifica o método utilizado.
        item["reranker_method"] = "lexical"

        results.append(item)

    results.sort(
        key=lambda x: x.get(
            "reranker_score",
            0.0,
        ),
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# CROSS ENCODER
# ============================================================

def _cross_encoder_rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int,
):
    """
    Reranking utilizando CrossEncoder.
    """

    model = _load_reranker()

    pairs = [
        (
            query,
            document.get(
                "content",
                "",
            ),
        )
        for document in documents
    ]

    scores = model.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )

    results = []

    for document, score in zip(
        documents,
        scores,
    ):

        item = dict(document)

        item["reranker_score"] = float(
            score
        )

        item["reranker_method"] = (
            "cross_encoder"
        )

        results.append(item)

    results.sort(
        key=lambda x: x.get(
            "reranker_score",
            float("-inf"),
        ),
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 5,
):
    """
    Reordena os documentos recuperados pelo RAG.

    Estratégia:

        Retriever
             ↓
        documentos
             ↓
        CrossEncoder
             ↓
        Top K

    Caso o modelo não esteja disponível:

        Retriever
             ↓
        Fallback lexical
             ↓
        Top K
    """

    if not documents:
        return []

    query = (query or "").strip()

    if not query:
        return documents[:top_k]

    # --------------------------------------------------------
    # Validação do TOP K
    # --------------------------------------------------------

    try:
        top_k = int(top_k)
    except (
        TypeError,
        ValueError,
    ):
        top_k = 5

    top_k = max(
        1,
        min(top_k, len(documents)),
    )

    # --------------------------------------------------------
    # Limita quantidade de candidatos
    # --------------------------------------------------------

    candidates = documents[:]

    # --------------------------------------------------------
    # CrossEncoder
    # --------------------------------------------------------

    try:

        return _cross_encoder_rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Fallback lexical
        # ----------------------------------------------------

        results = _lexical_rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )

        # Informação de diagnóstico.
        for item in results:

            item["reranker_error"] = (
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )

        return results


# ============================================================
# STATUS
# ============================================================

def reranker_status():
    """
    Retorna informações sobre o Reranker.

    Útil para:
        - Dashboard;
        - Configurações;
        - Health Check;
        - Auditoria.
    """

    try:

        model = _load_reranker()

        return {
            "configured": True,
            "model": RERANK_MODEL,
            "status": "ready",
            "type": "CrossEncoder",
            "batch_size": RERANK_BATCH_SIZE,
        }

    except Exception as exc:

        return {
            "configured": False,
            "model": RERANK_MODEL,
            "status": "fallback",
            "type": "lexical",
            "batch_size": RERANK_BATCH_SIZE,
            "error": (
                f"{type(exc).__name__}: "
                f"{str(exc)[:300]}"
            ),
        }


# ============================================================
# LIMPEZA DE CACHE
# ============================================================

def clear_reranker_cache():
    """
    Remove o modelo da memória.

    Útil quando:
        - o modelo for alterado;
        - houver manutenção;
        - testes;
        - troca de configuração.
    """

    _load_reranker.cache_clear()
