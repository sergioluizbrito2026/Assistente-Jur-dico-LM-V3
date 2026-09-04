"""
Assistente Jurídico SaaS IA V3.1
services/reranker.py

Reranker responsável por reordenar as evidências recuperadas
pelo mecanismo semântico.

Pipeline:

    Retriever
        ↓
    Candidatos
        ↓
    CrossEncoder
        ↓
    Ranking
        ↓
    Top K
        ↓
    RAG Pipeline
        ↓
    AI Service

Características:
- CrossEncoder com cache em memória.
- Fallback lexical automático.
- Normalização dos documentos.
- Controle seguro de TOP K.
- Controle de batch.
- Diagnóstico do erro do modelo.
- Compatibilidade com rag_pipeline.py.
- Não depende de Streamlit.
- Não depende diretamente do LLM.
- Suporte a scores negativos.
- Funções de status e manutenção.
- Self-test estrutural.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence
import os
import re


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
).strip()


def _get_batch_size() -> int:
    """
    Retorna o tamanho do batch de forma segura.
    """

    try:
        value = int(
            os.getenv(
                "RERANK_BATCH_SIZE",
                "16",
            )
        )
    except (TypeError, ValueError):
        value = 16

    return max(
        1,
        min(value, 128),
    )


RERANK_BATCH_SIZE = _get_batch_size()


def _get_max_candidates() -> int:
    """
    Quantidade máxima de documentos enviados ao CrossEncoder.
    """

    try:
        value = int(
            os.getenv(
                "RERANK_MAX_CANDIDATES",
                "30",
            )
        )
    except (TypeError, ValueError):
        value = 30

    return max(
        1,
        min(value, 100),
    )


RERANK_MAX_CANDIDATES = _get_max_candidates()


# ============================================================
# MODELO
# ============================================================

@lru_cache(maxsize=1)
def _load_reranker():
    """
    Carrega o CrossEncoder uma única vez por processo.

    O modelo permanece em memória para evitar
    carregamentos repetidos a cada pergunta.
    """

    if not RERANK_MODEL:
        raise RuntimeError(
            "RERANK_MODEL não configurado."
        )

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        RERANK_MODEL
    )


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def _normalize_text(
    text: Any,
) -> str:
    """
    Normaliza texto para comparação lexical.
    """

    if text is None:
        return ""

    try:
        text = str(text)
    except Exception:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokens(
    text: Any,
) -> set[str]:
    """
    Extrai tokens simples.

    Mantém suporte a caracteres acentuados
    utilizados em documentos jurídicos em português.
    """

    normalized = _normalize_text(
        text
    )

    if not normalized:
        return set()

    return set(
        re.findall(
            r"[a-zA-ZÀ-ÿ0-9]+",
            normalized,
        )
    )


# ============================================================
# NORMALIZAÇÃO DE DOCUMENTOS
# ============================================================

def _normalize_document(
    document: Any,
) -> Dict[str, Any] | None:
    """
    Normaliza documentos para o formato interno do reranker.

    Aceita:
        dict
        objetos com page_content
        objetos com content
        objetos com text
    """

    if document is None:
        return None

    # --------------------------------------------------------
    # Dict
    # --------------------------------------------------------

    if isinstance(
        document,
        dict,
    ):

        result = dict(
            document
        )

    # --------------------------------------------------------
    # Objetos LangChain / similares
    # --------------------------------------------------------

    else:

        result = {}

        if hasattr(
            document,
            "page_content",
        ):

            try:

                result["content"] = str(
                    getattr(
                        document,
                        "page_content",
                    )
                )

            except Exception:
                pass

        if hasattr(
            document,
            "metadata",
        ):

            try:

                metadata = getattr(
                    document,
                    "metadata",
                )

                if isinstance(
                    metadata,
                    dict,
                ):

                    result.update(
                        metadata
                    )

            except Exception:
                pass

        if not result:

            try:

                result["content"] = str(
                    document
                )

            except Exception:

                return None

    # --------------------------------------------------------
    # Descoberta do conteúdo
    # --------------------------------------------------------

    content = (
        result.get("content")
        or result.get("text")
        or result.get("page_content")
        or result.get("chunk")
        or result.get("document_content")
        or result.get("body")
        or ""
    )

    try:
        content = str(
            content
        ).strip()
    except Exception:
        content = ""

    if not content:
        return None

    result["content"] = content

    return result


def _normalize_documents(
    documents: Sequence[Any] | None,
) -> List[Dict[str, Any]]:
    """
    Normaliza uma coleção de documentos.
    """

    if not documents:
        return []

    normalized: List[
        Dict[str, Any]
    ] = []

    for document in documents:

        item = _normalize_document(
            document
        )

        if item is not None:

            normalized.append(
                item
            )

    return normalized


# ============================================================
# VALIDAÇÃO
# ============================================================

def _safe_top_k(
    value: Any,
    default: int,
    maximum: int,
) -> int:
    """
    Converte TOP K para inteiro seguro.
    """

    try:
        value = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        value = default

    return max(
        1,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# FALLBACK LEXICAL
# ============================================================

def _lexical_score(
    query_tokens: set[str],
    document_tokens: set[str],
) -> float:
    """
    Calcula uma pontuação lexical simples.

    Fórmula:

        tokens encontrados / tokens da pergunta

    Retorna valor entre 0 e 1.
    """

    if not query_tokens:
        return 0.0

    if not document_tokens:
        return 0.0

    intersection = (
        query_tokens
        & document_tokens
    )

    return float(
        len(intersection)
        / max(
            1,
            len(query_tokens),
        )
    )


def _lexical_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Fallback lexical utilizado quando o CrossEncoder
    não estiver disponível.
    """

    if not documents:
        return []

    query_tokens = _tokens(
        query
    )

    results: List[
        Dict[str, Any]
    ] = []

    for document in documents:

        content = document.get(
            "content",
            "",
        )

        document_tokens = _tokens(
            content
        )

        score = _lexical_score(
            query_tokens,
            document_tokens,
        )

        item = dict(
            document
        )

        item[
            "reranker_score"
        ] = float(
            score
        )

        item[
            "reranker_method"
        ] = "lexical"

        results.append(
            item
        )

    results.sort(
        key=lambda item: float(
            item.get(
                "reranker_score",
                0.0,
            )
        ),
        reverse=True,
    )

    return results[
        :top_k
    ]


# ============================================================
# CROSS ENCODER
# ============================================================

def _cross_encoder_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Executa o reranking utilizando CrossEncoder.
    """

    if not documents:
        return []

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

    results: List[
        Dict[str, Any]
    ] = []

    for document, score in zip(
        documents,
        scores,
    ):

        item = dict(
            document
        )

        try:

            item[
                "reranker_score"
            ] = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ):

            item[
                "reranker_score"
            ] = 0.0

        item[
            "reranker_method"
        ] = "cross_encoder"

        results.append(
            item
        )

    results.sort(
        key=lambda item: float(
            item.get(
                "reranker_score",
                float("-inf"),
            )
        ),
        reverse=True,
    )

    return results[
        :top_k
    ]


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def rerank(
    query: str,
    documents: Sequence[Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Reordena documentos recuperados pelo RAG.

    Estratégia:

        Retriever
             ↓
        candidatos
             ↓
        CrossEncoder
             ↓
        ranking
             ↓
        Top K

    Caso o CrossEncoder falhe:

        Retriever
             ↓
        fallback lexical
             ↓
        Top K
    """

    # --------------------------------------------------------
    # Normalização
    # --------------------------------------------------------

    normalized = _normalize_documents(
        documents
    )

    if not normalized:
        return []

    query = _normalize_text(
        query
    )

    if not query:
        return normalized[
            :max(
                1,
                min(
                    int(top_k)
                    if str(top_k).isdigit()
                    else 5,
                    len(normalized),
                ),
            )
        ]

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_k = _safe_top_k(
        top_k,
        default=5,
        maximum=len(normalized),
    )

    # --------------------------------------------------------
    # Limite de candidatos
    # --------------------------------------------------------

    candidates = normalized[
        :RERANK_MAX_CANDIDATES
    ]

    # --------------------------------------------------------
    # CrossEncoder
    # --------------------------------------------------------

    try:

        results = _cross_encoder_rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )

        return results

    except Exception as exc:

        # ----------------------------------------------------
        # Fallback lexical
        # ----------------------------------------------------

        results = _lexical_rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )

        error_message = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:300]}"
        )

        for item in results:

            item[
                "reranker_error"
            ] = error_message

        return results


# ============================================================
# STATUS
# ============================================================

def reranker_status(
    load_model: bool = False,
) -> Dict[str, Any]:
    """
    Retorna o estado do Reranker.

    Por padrão não carrega o modelo, evitando que o Dashboard
    provoque download/carregamento pesado.

    Para testar efetivamente o CrossEncoder:

        reranker_status(load_model=True)
    """

    status: Dict[str, Any] = {
        "configured": bool(
            RERANK_MODEL
        ),
        "model": RERANK_MODEL,
        "status": "configured",
        "type": "CrossEncoder",
        "batch_size": RERANK_BATCH_SIZE,
        "max_candidates": RERANK_MAX_CANDIDATES,
    }

    if not RERANK_MODEL:

        status[
            "configured"
        ] = False

        status[
            "status"
        ] = "not_configured"

        status[
            "type"
        ] = "lexical"

        return status

    if not load_model:

        return status

    try:

        _load_reranker()

        status[
            "status"
        ] = "ready"

    except Exception as exc:

        status[
            "configured"
        ] = False

        status[
            "status"
        ] = "fallback"

        status[
            "type"
        ] = "lexical"

        status[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:300]}"
        )

    return status


# ============================================================
# LIMPEZA DE CACHE
# ============================================================

def clear_reranker_cache():
    """
    Remove o CrossEncoder da memória.
    """

    _load_reranker.cache_clear()


# ============================================================
# TESTE LEXICAL
# ============================================================

def lexical_self_test() -> Dict[str, Any]:
    """
    Testa o fallback lexical sem carregar modelo.
    """

    query = (
        "prazo para contestação"
    )

    documents = [
        {
            "chunk_id": "1",
            "document": "teste.pdf",
            "content": (
                "O prazo para contestação "
                "é de quinze dias úteis."
            ),
        },
        {
            "chunk_id": "2",
            "document": "outro.pdf",
            "content": (
                "O contrato possui "
                "cláusula de confidencialidade."
            ),
        },
    ]

    results = _lexical_rerank(
        query=query,
        documents=documents,
        top_k=2,
    )

    return {
        "status": "ok",
        "results_count": len(
            results
        ),
        "top_chunk": (
            results[0].get(
                "chunk_id"
            )
            if results
            else None
        ),
        "method": (
            results[0].get(
                "reranker_method"
            )
            if results
            else None
        ),
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural do módulo.

    Não carrega o CrossEncoder.
    """

    required_functions = [
        "_load_reranker",
        "rerank",
        "reranker_status",
        "clear_reranker_cache",
        "lexical_self_test",
    ]

    missing = [
        name
        for name in required_functions
        if name not in globals()
    ]

    lexical = lexical_self_test()

    return {
        "valid": not missing,
        "module": "services.reranker",
        "required_functions": (
            required_functions
        ),
        "missing_functions": missing,
        "lexical_test": lexical,
        "model": RERANK_MODEL,
        "batch_size": RERANK_BATCH_SIZE,
        "max_candidates": (
            RERANK_MAX_CANDIDATES
        ),
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print(
        "RERANKER.PY V3.1 - SELF TEST"
    )
    print("=" * 60)

    print(
        "Status              : "
        f"{'OK' if result['valid'] else 'ERRO'}"
    )

    print(
        "Modelo              : "
        f"{result['model']}"
    )

    print(
        "Batch size          : "
        f"{result['batch_size']}"
    )

    print(
        "Max candidates      : "
        f"{result['max_candidates']}"
    )

    print(
        "Funções ausentes    : "
        f"{result['missing_functions']}"
    )

    print(
        "Teste lexical       : "
        f"{result['lexical_test']['status']}"
    )

    print(
        "Top chunk lexical   : "
        f"{result['lexical_test']['top_chunk']}"
    )

    print(
        "Método lexical      : "
        f"{result['lexical_test']['method']}"
    )

    print("=" * 60)
