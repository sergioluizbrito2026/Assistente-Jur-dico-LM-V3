from typing import Any

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

# Score mínimo do reranker para considerar uma evidência
# relevante. Pode ser ajustado posteriormente através
# das configurações do sistema.
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
# RECUPERAÇÃO + RERANKING
# ============================================================

def retrieve_and_rerank(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
):
    """
    Executa:

        Query
          ↓
        Retriever / Embeddings
          ↓
        FAISS
          ↓
        CrossEncoder
          ↓
        Top K final

    Mantém a estrutura de retorno compatível
    com o aplicativo atual.
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

    retrieved = semantic_search(
        query=query,
        org_id=org_id,
        top_k=top_k,
    )

    if not retrieved:

        return {
            "retrieved": [],
            "reranked": [],
        }

    # --------------------------------------------------------
    # 2. Reranking
    # --------------------------------------------------------

    reranked = rerank(
        query=query,
        documents=retrieved,
        top_k=rerank_k,
    )

    return {
        "retrieved": retrieved,
        "reranked": reranked,
    }


# ============================================================
# FILTRO DE EVIDÊNCIAS
# ============================================================

def _filter_evidence(chunks):
    """
    Remove evidências inválidas ou excessivamente fracas.

    O objetivo é impedir que resultados ruins sejam enviados
    ao LLM.
    """

    valid = []

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        content = (
            chunk.get("content")
            or ""
        ).strip()

        if not content:
            continue

        score = chunk.get(
            "reranker_score"
        )

        # ----------------------------------------------------
        # Se houver score, verifica limite.
        # ----------------------------------------------------

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

        valid.append(chunk)

    return valid


# ============================================================
# CONTEXTO CONTROLADO
# ============================================================

def _build_context(
    chunks,
    extra_context="",
):
    """
    Monta o contexto que será enviado ao LLM.

    Existe limite de caracteres para evitar prompts
    excessivamente grandes.
    """

    context = []

    # --------------------------------------------------------
    # Texto fornecido diretamente pelo usuário
    # --------------------------------------------------------

    if extra_context:

        extra_context = (
            extra_context
            or ""
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

    for chunk in chunks:

        context.append(chunk)

    # --------------------------------------------------------
    # Limite de contexto
    # --------------------------------------------------------

    final_context = []

    current_size = 0

    for chunk in context:

        content = (
            chunk.get(
                "content",
                "",
            )
            or ""
        )

        content_size = len(content)

        if (
            current_size + content_size
            > MAX_CONTEXT_CHARS
        ):
            break

        final_context.append(chunk)

        current_size += content_size

    return final_context


# ============================================================
# CITAÇÕES
# ============================================================

def _build_citations(context):
    """
    Cria referências estruturadas para a resposta.

    As posições [1], [2], [3] correspondem exatamente
    à ordem do contexto enviado ao LLM.
    """

    citations = []

    for index, chunk in enumerate(
        context,
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
            }
        )

    return citations


# ============================================================
# RESULTADO SEM EVIDÊNCIA
# ============================================================

def _empty_result():
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
    }


# ============================================================
# RAG PRINCIPAL
# ============================================================

def rag_answer(
    query,
    org_id,
    top_k=DEFAULT_TOP_K,
    rerank_k=DEFAULT_RERANK_K,
    extra_context="",
    generate_answer_flag=True,
):
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

    Mantém compatibilidade com o app.py atual.
    """

    query = (query or "").strip()

    if not query:

        return _empty_result()

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
            )

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
                    "type": type(exc).__name__,
                    "message": str(exc)[:300],
                },
            }

    # --------------------------------------------------------
    # 8. Resultado final
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
    }


# ============================================================
# HELPERS PARA FUTUROS AGENTES
# ============================================================

def build_agent_context(
    query,
    org_id,
    top_k=DEFAULT_TOP_K,
    rerank_k=DEFAULT_RERANK_K,
    extra_context="",
):
    """
    Executa apenas a parte RAG.

    Útil para os futuros agentes:

        Legal Agent
        Risk Agent
        Summary Agent

    Eles poderão reutilizar as mesmas evidências
    sem executar múltiplas buscas desnecessariamente.
    """

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
