"""
Assistente Jurídico SaaS IA V3
services/rag_pipeline.py

Pipeline principal de RAG.

Fluxo:

    Pergunta
       ↓
    Embeddings / Retriever
       ↓
    Reranker
       ↓
    Filtro de evidências
       ↓
    Contexto controlado
       ↓
    LLM
       ↓
    Citações
       ↓
    Resultado estruturado

Características:
- Compatível com o app.py V3.
- Compatível com services.embeddings.
- Compatível com services.reranker.
- Compatível com services.ai.
- Não depende de Streamlit.
- Trata erros de recuperação.
- Trata erros do reranker.
- Limita o tamanho do contexto.
- Gera citações estruturadas.
- Permite contexto adicional.
- Mantém funções auxiliares para futuros agentes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from services.ai import generate_answer
from services.embeddings import semantic_search
from services.reranker import rerank


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEFAULT_TOP_K = 10
DEFAULT_RERANK_K = 5

MAX_TOP_K = 30
MAX_RERANK_K = 10

MAX_CONTEXT_CHARS = 18000

# Score mínimo do reranker.
#
# Importante:
# CrossEncoder pode produzir scores negativos dependendo
# do modelo utilizado. Por isso o limite padrão é permissivo.
MIN_RERANK_SCORE = -5.0


# ============================================================
# VALIDAÇÃO
# ============================================================

def _safe_int(
    value: Any,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """
    Converte um valor para inteiro dentro de um intervalo seguro.
    """

    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default

    return max(
        minimum,
        min(value, maximum),
    )


# ============================================================
# NORMALIZAÇÃO DE CHUNK
# ============================================================

def _normalize_chunk(chunk: Any) -> Dict[str, Any] | None:
    """
    Normaliza diferentes formatos de chunks para um formato
    compatível com o restante do pipeline.

    Aceita:
        dict
        objetos com page_content
        objetos com content
        objetos com text
    """

    if chunk is None:
        return None

    if isinstance(chunk, dict):
        result = dict(chunk)

    else:
        result = {}

        # page_content
        if hasattr(chunk, "page_content"):
            try:
                result["content"] = str(
                    getattr(chunk, "page_content")
                )
            except Exception:
                pass

        # metadata
        if hasattr(chunk, "metadata"):
            try:
                metadata = getattr(chunk, "metadata")

                if isinstance(metadata, dict):
                    result.update(metadata)

            except Exception:
                pass

        # fallback
        if not result:
            try:
                result["content"] = str(chunk)
            except Exception:
                return None

    # Normaliza o conteúdo.
    content = (
        result.get("content")
        or result.get("text")
        or result.get("page_content")
        or result.get("chunk")
        or result.get("document_content")
        or result.get("body")
        or ""
    )

    result["content"] = str(content).strip()

    if not result["content"]:
        return None

    return result


def _normalize_chunks(
    chunks: Sequence[Any] | None,
) -> List[Dict[str, Any]]:
    """
    Normaliza uma coleção de chunks.
    """

    if not chunks:
        return []

    normalized: List[Dict[str, Any]] = []

    for chunk in chunks:

        item = _normalize_chunk(chunk)

        if item is not None:
            normalized.append(item)

    return normalized


# ============================================================
# RECUPERAÇÃO + RERANKING
# ============================================================

def retrieve_and_rerank(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
) -> Dict[str, Any]:
    """
    Executa:

        Query
          ↓
        Retriever / Embeddings
          ↓
        FAISS / índice semântico
          ↓
        CrossEncoder
          ↓
        Top K final

    Retorna:

        {
            "retrieved": [...],
            "reranked": [...]
        }
    """

    query = (query or "").strip()

    if not query:
        return {
            "retrieved": [],
            "reranked": [],
        }

    top_k = _safe_int(
        top_k,
        DEFAULT_TOP_K,
        1,
        MAX_TOP_K,
    )

    rerank_k = _safe_int(
        rerank_k,
        DEFAULT_RERANK_K,
        1,
        min(MAX_RERANK_K, top_k),
    )

    # --------------------------------------------------------
    # 1. Recuperação semântica
    # --------------------------------------------------------

    try:

        retrieved = semantic_search(
            query=query,
            org_id=org_id,
            top_k=top_k,
        )

    except TypeError:

        # Compatibilidade com versões diferentes
        # do embeddings.py.
        try:

            retrieved = semantic_search(
                query,
                org_id,
                top_k,
            )

        except Exception:
            retrieved = []

    except Exception:
        retrieved = []

    retrieved = _normalize_chunks(
        retrieved
    )

    if not retrieved:

        return {
            "retrieved": [],
            "reranked": [],
        }

    # --------------------------------------------------------
    # 2. Reranking
    # --------------------------------------------------------

    try:

        reranked = rerank(
            query=query,
            documents=retrieved,
            top_k=rerank_k,
        )

    except TypeError:

        # Compatibilidade com versões diferentes
        # do reranker.py.
        try:

            reranked = rerank(
                query,
                retrieved,
                rerank_k,
            )

        except Exception:
            reranked = retrieved[:rerank_k]

    except Exception:

        # Se o reranker falhar, não derruba o sistema.
        # Usa os resultados recuperados como fallback.
        reranked = retrieved[:rerank_k]

    reranked = _normalize_chunks(
        reranked
    )

    if not reranked:
        reranked = retrieved[:rerank_k]

    return {
        "retrieved": retrieved,
        "reranked": reranked,
    }


# ============================================================
# FILTRO DE EVIDÊNCIAS
# ============================================================

def _filter_evidence(
    chunks: Sequence[Any] | None,
) -> List[Dict[str, Any]]:
    """
    Remove evidências inválidas ou excessivamente fracas.

    O objetivo é impedir que resultados ruins sejam enviados
    ao LLM.
    """

    if not chunks:
        return []

    valid: List[Dict[str, Any]] = []

    for chunk in chunks:

        normalized = _normalize_chunk(
            chunk
        )

        if normalized is None:
            continue

        content = (
            normalized.get("content")
            or ""
        ).strip()

        if not content:
            continue

        score = normalized.get(
            "reranker_score"
        )

        if score is not None:

            try:

                score = float(score)

                if score < MIN_RERANK_SCORE:
                    continue

            except (
                TypeError,
                ValueError,
            ):
                pass

        valid.append(
            normalized
        )

    return valid


# ============================================================
# CONTEXTO CONTROLADO
# ============================================================

def _build_context(
    chunks: Sequence[Any] | None,
    extra_context: str = "",
) -> List[Dict[str, Any]]:
    """
    Monta o contexto que será enviado ao LLM.

    Existe limite de caracteres para evitar prompts
    excessivamente grandes.
    """

    context: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Texto fornecido diretamente pelo usuário
    # --------------------------------------------------------

    extra_context = (
        extra_context or ""
    ).strip()

    if extra_context:

        context.append(
            {
                "chunk_id": "user_input",
                "document": "Texto fornecido pelo usuário",
                "document_id": None,
                "organization_id": None,
                "content": extra_context,
                "page": "N/D",
                "chunk_index": None,
                "reranker_score": 1.0,
                "retriever_score": 1.0,
                "reranker_method": "user_input",
            }
        )

    # --------------------------------------------------------
    # Evidências recuperadas
    # --------------------------------------------------------

    for chunk in chunks or []:

        normalized = _normalize_chunk(
            chunk
        )

        if normalized is None:
            continue

        context.append(
            normalized
        )

    # --------------------------------------------------------
    # Limite de contexto
    # --------------------------------------------------------

    final_context: List[Dict[str, Any]] = []

    current_size = 0

    for chunk in context:

        content = (
            chunk.get(
                "content",
                "",
            )
            or ""
        )

        content = str(content)

        content_size = len(
            content
        )

        # Permite pelo menos um contexto,
        # mesmo que ele seja maior que o limite.
        if (
            current_size + content_size
            > MAX_CONTEXT_CHARS
        ):

            if not final_context:

                truncated = dict(chunk)

                truncated["content"] = content[
                    :MAX_CONTEXT_CHARS
                ]

                final_context.append(
                    truncated
                )

            break

        final_context.append(
            chunk
        )

        current_size += content_size

    return final_context


# ============================================================
# CITAÇÕES
# ============================================================

def _build_citations(
    context: Sequence[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """
    Cria referências estruturadas para a resposta.

    As posições [1], [2], [3] correspondem exatamente
    à ordem do contexto enviado ao LLM.
    """

    citations: List[Dict[str, Any]] = []

    for index, chunk in enumerate(
        context or [],
        start=1,
    ):

        citations.append(
            {
                "id": index,

                "document": chunk.get(
                    "document",
                    "Desconhecido",
                ),

                "document_id": chunk.get(
                    "document_id"
                ),

                "page": chunk.get(
                    "page",
                    "N/D",
                ),

                "chunk_id": chunk.get(
                    "chunk_id"
                ),

                "chunk_index": chunk.get(
                    "chunk_index"
                ),

                "retriever_score": chunk.get(
                    "retriever_score"
                ),

                "reranker_score": chunk.get(
                    "reranker_score"
                ),

                "reranker_method": chunk.get(
                    "reranker_method"
                ),

                "content": chunk.get(
                    "content",
                    "",
                ),
            }
        )

    return citations


# ============================================================
# RESULTADO SEM EVIDÊNCIA
# ============================================================

def _empty_result(
    reason: str = "Evidência insuficiente.",
) -> Dict[str, Any]:
    """
    Retorno padronizado quando não existem evidências.
    """

    return {
        "answer": "",
        "retrieved": [],
        "reranked": [],
        "context": [],
        "citations": [],
        "evidence_status": "insufficient",
        "evidence_count": 0,
        "answer_generated": False,
        "error": None,
        "reason": reason,
    }


# ============================================================
# RAG PRINCIPAL
# ============================================================

def rag_answer(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    extra_context: str = "",
    generate_answer_flag: bool = True,
    agent_instruction: str | None = None,
) -> Dict[str, Any]:
    """
    Pipeline RAG completo:

        Pergunta
            ↓
        Retriever
            ↓
        Reranker
            ↓
        Evidence Filter
            ↓
        Context Builder
            ↓
        LLM
            ↓
        Citações
            ↓
        Resultado estruturado

    Mantém compatibilidade com o app.py V3.
    """

    query = (query or "").strip()

    if not query:
        return _empty_result(
            "Pergunta vazia."
        )

    # --------------------------------------------------------
    # 1. Retrieval + Reranking
    # --------------------------------------------------------

    result = retrieve_and_rerank(
        query=query,
        org_id=org_id,
        top_k=top_k,
        rerank_k=rerank_k,
    )

    retrieved = result.get(
        "retrieved",
        [],
    )

    reranked = result.get(
        "reranked",
        [],
    )

    # --------------------------------------------------------
    # 2. Filtro de evidências
    # --------------------------------------------------------

    evidence = _filter_evidence(
        reranked
    )

    # --------------------------------------------------------
    # 3. Montagem do contexto
    # --------------------------------------------------------

    context = _build_context(
        evidence,
        extra_context=extra_context,
    )

    # --------------------------------------------------------
    # 4. Citações
    # --------------------------------------------------------

    citations = _build_citations(
        context
    )

    # --------------------------------------------------------
    # 5. Verificação de evidência
    # --------------------------------------------------------

    has_document_evidence = bool(
        evidence
    )

    has_extra_context = bool(
        (extra_context or "").strip()
    )

    has_evidence = (
        has_document_evidence
        or has_extra_context
    )

    # --------------------------------------------------------
    # 6. Sem evidência e sem contexto do usuário
    # --------------------------------------------------------

    if (
        not has_evidence
        and generate_answer_flag
    ):

        return {
            "answer": (
                "Não há evidência suficiente "
                "nos documentos disponibilizados "
                "para responder com segurança."
            ),
            "retrieved": retrieved,
            "reranked": reranked,
            "context": [],
            "citations": [],
            "evidence_status": "insufficient",
            "evidence_count": 0,
            "answer_generated": False,
            "error": None,
        }

    # --------------------------------------------------------
    # 7. Geração da resposta
    # --------------------------------------------------------

    answer = ""

    if (
        generate_answer_flag
        and callable(generate_answer)
    ):

        try:

            answer = generate_answer(
                query,
                context,
                agent_instruction=agent_instruction,
            )

            # Algumas implementações de IA podem retornar
            # None ou objetos não-string.
            if answer is None:
                answer = ""

            else:
                answer = str(
                    answer
                ).strip()

        except TypeError:

            # Compatibilidade com uma versão de
            # generate_answer que não aceita
            # agent_instruction.
            try:

                answer = generate_answer(
                    query,
                    context,
                )

                if answer is None:
                    answer = ""

                else:
                    answer = str(
                        answer
                    ).strip()

            except Exception as exc:

                answer = (
                    "Não foi possível gerar a resposta "
                    "da IA neste momento."
                )

                return {
                    "answer": answer,
                    "retrieved": retrieved,
                    "reranked": reranked,
                    "context": context,
                    "citations": citations,
                    "evidence_status": (
                        "available"
                        if has_evidence
                        else "insufficient"
                    ),
                    "evidence_count": len(
                        context
                    ),
                    "answer_generated": False,
                    "error": {
                        "type": type(
                            exc
                        ).__name__,
                        "message": str(
                            exc
                        )[:300],
                    },
                }

        except Exception as exc:

            answer = (
                "Não foi possível gerar a resposta "
                "da IA neste momento."
            )

            return {
                "answer": answer,
                "retrieved": retrieved,
                "reranked": reranked,
                "context": context,
                "citations": citations,
                "evidence_status": (
                    "available"
                    if has_evidence
                    else "insufficient"
                ),
                "evidence_count": len(
                    context
                ),
                "answer_generated": False,
                "error": {
                    "type": type(
                        exc
                    ).__name__,
                    "message": str(
                        exc
                    )[:300],
                },
            }

    # --------------------------------------------------------
    # 8. Quando geração está desativada
    # --------------------------------------------------------

    elif not generate_answer_flag:

        answer = ""

    # --------------------------------------------------------
    # 9. Resultado final
    # --------------------------------------------------------

    return {
        "answer": answer,

        "retrieved": retrieved,

        "reranked": reranked,

        "context": context,

        "citations": citations,

        "evidence_status": (
            "available"
            if has_evidence
            else "insufficient"
        ),

        "evidence_count": len(
            context
        ),

        "answer_generated": bool(
            answer
        ),

        "error": None,
    }


# ============================================================
# HELPERS PARA FUTUROS AGENTES
# ============================================================

def build_agent_context(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    Executa apenas a parte RAG.

    Útil para futuros agentes:

        Legal Agent
        Risk Agent
        Summary Agent

    Eles poderão reutilizar as mesmas evidências
    sem executar múltiplas buscas desnecessariamente.
    """

    query = (query or "").strip()

    if not query:

        return {
            "query": query,
            "org_id": org_id,
            "retrieved": [],
            "reranked": [],
            "context": [],
            "citations": [],
            "evidence_status": "insufficient",
            "evidence_count": 0,
        }

    result = retrieve_and_rerank(
        query=query,
        org_id=org_id,
        top_k=top_k,
        rerank_k=rerank_k,
    )

    evidence = _filter_evidence(
        result.get(
            "reranked",
            [],
        )
    )

    context = _build_context(
        evidence,
        extra_context=extra_context,
    )

    citations = _build_citations(
        context
    )

    return {
        "query": query,

        "org_id": org_id,

        "retrieved": result.get(
            "retrieved",
            [],
        ),

        "reranked": result.get(
            "reranked",
            [],
        ),

        "context": context,

        "citations": citations,

        "evidence_status": (
            "available"
            if context
            else "insufficient"
        ),

        "evidence_count": len(
            context
        ),
    }


# ============================================================
# ALIASES
# ============================================================

def run_rag(
    query: str,
    org_id: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Alias adicional para integração futura.
    """

    return rag_answer(
        query=query,
        org_id=org_id,
        **kwargs,
    )


# ============================================================
# TESTE INTERNO
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural do módulo.

    Não executa embeddings, FAISS, reranker ou LLM.
    """

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "document": "teste.pdf",
            "page": 1,
            "content": (
                "O prazo processual para contestação "
                "é de quinze dias úteis."
            ),
            "retriever_score": 0.90,
            "reranker_score": 0.92,
        }
    ]

    filtered = _filter_evidence(
        chunks
    )

    context = _build_context(
        filtered
    )

    citations = _build_citations(
        context
    )

    return {
        "module": "rag_pipeline.py",
        "status": "ok",
        "filtered_count": len(
            filtered
        ),
        "context_count": len(
            context
        ),
        "citation_count": len(
            citations
        ),
        "has_rag_answer": callable(
            rag_answer
        ),
        "has_retrieve_and_rerank": callable(
            retrieve_and_rerank
        ),
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print("RAG_PIPELINE.PY V3 - SELF TEST")
    print("=" * 60)

    print(
        f"Status              : "
        f"{result['status']}"
    )

    print(
        f"Filtered chunks     : "
        f"{result['filtered_count']}"
    )

    print(
        f"Contextos           : "
        f"{result['context_count']}"
    )

    print(
        f"Citações            : "
        f"{result['citation_count']}"
    )

    print(
        f"rag_answer          : "
        f"{result['has_rag_answer']}"
    )

    print(
        f"retrieve_and_rerank : "
        f"{result['has_retrieve_and_rerank']}"
    )

    print("=" * 60)
