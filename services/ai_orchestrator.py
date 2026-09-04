"""
Assistente Jurídico SaaS IA V3.1
services/ai_orchestrator.py

Orquestrador central dos agentes jurídicos.

Agentes suportados:
    1. Agente Jurídico
    2. Agente de Risco
    3. Agente de Resumo
    4. Agente Geral
    5. RAG / Recuperação de documentos
    6. Citações e evidências
    7. Evaluation / métricas
    8. Guard Agent

Objetivos:
- Centralizar a execução dos agentes.
- Não derrubar o Streamlit quando um agente falhar.
- Aceitar diferentes formatos de retorno dos agentes.
- Usar o RAG como fonte principal de evidências.
- Permitir fallback para respostas disponíveis.
- Manter compatibilidade com app.py V3.
- Não depender diretamente do Streamlit.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Sequence


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# IMPORTS SEGUROS
# ============================================================

try:
    from services.ai import generate_answer
except Exception:
    generate_answer = None


try:
    from services.rag_pipeline import (
        rag_answer,
        retrieve_and_rerank,
    )
except Exception:
    rag_answer = None
    retrieve_and_rerank = None


try:
    from services.evaluation import evaluate_answer
except Exception:
    evaluate_answer = None


# ============================================================
# UTILITÁRIOS
# ============================================================

def _safe_text(value: Any) -> str:
    """Converte qualquer valor em texto com segurança."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _safe_list(value: Any) -> List[Any]:
    """Converte valores possíveis em lista."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


def _safe_dict(value: Any) -> Dict[str, Any]:
    """Converte retorno para dicionário."""

    if isinstance(value, dict):
        return value

    return {}


def _extract_answer(result: Any) -> str:
    """
    Extrai texto de diferentes formatos de retorno.

    Aceita:

        "texto"

    ou:

        {"answer": "texto"}

    ou:

        {"response": "texto"}

    ou:

        {"text": "texto"}

    ou:

        {"content": "texto"}
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if not isinstance(result, dict):
        return _safe_text(result)

    for field in (
        "answer",
        "response",
        "text",
        "content",
        "output",
        "result",
        "message",
    ):
        value = result.get(field)

        if value:
            return _safe_text(value)

    return ""


def _extract_chunks(result: Any) -> List[Any]:
    """Extrai chunks de diferentes estruturas."""

    if not isinstance(result, dict):
        return []

    for field in (
        "reranked",
        "chunks",
        "results",
        "documents",
        "context",
        "retrieved",
    ):
        value = result.get(field)

        if value:
            return _safe_list(value)

    return []


def _extract_citations(result: Any) -> List[Any]:
    """Extrai citações de diferentes estruturas."""

    if not isinstance(result, dict):
        return []

    for field in (
        "citations",
        "references",
        "sources",
        "evidence",
    ):
        value = result.get(field)

        if value:
            return _safe_list(value)

    return []


def _error(message: str, exc: Exception | None = None) -> Dict[str, Any]:
    """Cria resposta padronizada de erro."""

    if exc is not None:
        logger.exception(message)

    return {
        "success": False,
        "answer": "",
        "error": message,
    }


# ============================================================
# AGENTE JURÍDICO
# ============================================================

def legal_agent(
    question: str,
    context: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Agente Jurídico.

    Produz resposta jurídica baseada nas evidências fornecidas.
    """

    question = _safe_text(question)

    if not question:
        return _error("Pergunta vazia.")

    context = list(context or [])
    citations = list(citations or [])

    prompt = f"""
Você é o Agente Jurídico de uma plataforma de IA jurídica.

Responda à pergunta utilizando SOMENTE as evidências fornecidas.

REGRAS:
- Não invente informações.
- Não invente artigos de lei.
- Não invente jurisprudência.
- Não invente fatos.
- Se as evidências não forem suficientes, informe isso claramente.
- Seja objetivo.
- Cite as evidências quando disponíveis.

PERGUNTA:
{question}

EVIDÊNCIAS:
{_format_context(context)}

CITAÇÕES:
{_format_citations(citations)}
"""

    if generate_answer is None:
        return _error(
            "Serviço de IA não está disponível."
        )

    try:
        result = generate_answer(prompt)

        answer = _extract_answer(result)

        if not answer:
            return _error(
                "O Agente Jurídico não retornou uma resposta."
            )

        return {
            "success": True,
            "answer": answer,
            "agent": "legal",
            "raw": result,
        }

    except Exception as exc:
        return _error(
            "Falha no Agente Jurídico.",
            exc,
        )


# ============================================================
# AGENTE DE RISCO
# ============================================================

def risk_analysis(
    question: str,
    context: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Agente de Risco.

    Identifica riscos jurídicos presentes nas evidências.
    """

    question = _safe_text(question)

    context = list(context or [])
    citations = list(citations or [])

    prompt = f"""
Você é o Agente de Risco Jurídico.

Analise somente as evidências fornecidas.

Identifique:
1. Riscos jurídicos.
2. Pontos de atenção.
3. Possíveis inconsistências.
4. Informações ausentes.
5. Grau de atenção.

Não invente informações.

PERGUNTA:
{question}

EVIDÊNCIAS:
{_format_context(context)}

CITAÇÕES:
{_format_citations(citations)}
"""

    if generate_answer is None:
        return _error(
            "Serviço de IA não está disponível."
        )

    try:
        result = generate_answer(prompt)

        answer = _extract_answer(result)

        return {
            "success": bool(answer),
            "answer": answer,
            "agent": "risk",
            "raw": result,
        }

    except Exception as exc:
        return _error(
            "Falha no Agente de Risco.",
            exc,
        )


# ============================================================
# AGENTE DE RESUMO
# ============================================================

def summary_agent(
    question: str,
    context: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Agente de Resumo.
    """

    question = _safe_text(question)
    context = list(context or [])

    prompt = f"""
Você é o Agente de Resumo Jurídico.

Resuma somente as informações existentes nas evidências.

Seja:
- objetivo;
- claro;
- organizado;
- fiel ao documento.

PERGUNTA:
{question}

DOCUMENTOS:
{_format_context(context)}
"""

    if generate_answer is None:
        return _error(
            "Serviço de IA não está disponível."
        )

    try:
        result = generate_answer(prompt)

        answer = _extract_answer(result)

        return {
            "success": bool(answer),
            "answer": answer,
            "agent": "summary",
            "raw": result,
        }

    except Exception as exc:
        return _error(
            "Falha no Agente de Resumo.",
            exc,
        )


# ============================================================
# AGENTE GERAL
# ============================================================

def general_agent(
    question: str,
    context: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Agente Geral.
    """

    question = _safe_text(question)

    prompt = f"""
Você é o Agente Geral do Assistente Jurídico SaaS IA.

Responda de maneira clara e objetiva.

Utilize as evidências disponíveis.

Não invente informações jurídicas.

Se não houver evidência suficiente, diga explicitamente
que não é possível responder com segurança.

PERGUNTA:
{question}

EVIDÊNCIAS:
{_format_context(context or [])}

CITAÇÕES:
{_format_citations(citations or [])}
"""

    if generate_answer is None:
        return _error(
            "Serviço de IA não está disponível."
        )

    try:
        result = generate_answer(prompt)

        answer = _extract_answer(result)

        return {
            "success": bool(answer),
            "answer": answer,
            "agent": "general",
            "raw": result,
        }

    except Exception as exc:
        return _error(
            "Falha no Agente Geral.",
            exc,
        )


# ============================================================
# RAG
# ============================================================

def _run_rag(
    question: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Executa recuperação RAG.

    Primeiro tenta retrieve_and_rerank.
    Depois tenta rag_answer.
    """

    if not question:
        return {
            "success": False,
            "chunks": [],
            "citations": [],
            "answer": "",
        }

    # --------------------------------------------------------
    # Recuperação + reranking
    # --------------------------------------------------------

    if retrieve_and_rerank is not None:

        try:

            result = retrieve_and_rerank(
                question,
                top_k=top_k,
            )

            chunks = _extract_chunks(result)
            citations = _extract_citations(result)

            return {
                "success": bool(chunks),
                "chunks": chunks,
                "citations": citations,
                "answer": _extract_answer(result),
                "raw": result,
            }

        except TypeError:

            try:

                result = retrieve_and_rerank(
                    question
                )

                chunks = _extract_chunks(result)
                citations = _extract_citations(result)

                return {
                    "success": bool(chunks),
                    "chunks": chunks,
                    "citations": citations,
                    "answer": _extract_answer(result),
                    "raw": result,
                }

            except Exception as exc:
                logger.exception(
                    "Erro na recuperação RAG."
                )

        except Exception:
            logger.exception(
                "Erro na recuperação RAG."
            )

    # --------------------------------------------------------
    # RAG answer direto
    # --------------------------------------------------------

    if rag_answer is not None:

        try:

            result = rag_answer(
                question
            )

            chunks = _extract_chunks(result)
            citations = _extract_citations(result)
            answer = _extract_answer(result)

            return {
                "success": bool(answer or chunks),
                "chunks": chunks,
                "citations": citations,
                "answer": answer,
                "raw": result,
            }

        except Exception:
            logger.exception(
                "Erro no rag_answer."
            )

    return {
        "success": False,
        "chunks": [],
        "citations": [],
        "answer": "",
    }


# ============================================================
# GUARD AGENT
# ============================================================

def guard_agent(
    question: str,
    answer: str,
    context: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Guard Agent determinístico.

    Verifica condições básicas de segurança.

    Não utiliza LLM.
    """

    question = _safe_text(question)
    answer = _safe_text(answer)
    context = list(context or [])

    issues: List[str] = []

    if not question:
        issues.append(
            "Pergunta vazia."
        )

    if not answer:
        issues.append(
            "Resposta vazia."
        )

    if not context:
        issues.append(
            "Nenhuma evidência recuperada."
        )

    # Indicadores de possível afirmação sem evidência.
    suspicious_phrases = (
        "tenho certeza absoluta",
        "garanto que",
        "com certeza",
        "sem qualquer dúvida",
    )

    answer_lower = answer.lower()

    for phrase in suspicious_phrases:

        if phrase in answer_lower:

            issues.append(
                "A resposta contém linguagem de certeza excessiva."
            )

            break

    return {
        "success": len(issues) == 0,
        "approved": len(issues) == 0,
        "issues": issues,
        "agent": "guard",
    }


# ============================================================
# FORMATAÇÃO
# ============================================================

def _format_context(
    chunks: Sequence[Any],
) -> str:
    """Formata evidências para o prompt."""

    if not chunks:
        return "Nenhuma evidência recuperada."

    output: List[str] = []

    for index, chunk in enumerate(chunks, start=1):

        if isinstance(chunk, dict):

            content = (
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("page_content")
                or ""
            )

            document = (
                chunk.get("document")
                or chunk.get("document_name")
                or chunk.get("name")
                or "Documento"
            )

            page = chunk.get("page")

            if page is not None:
                location = f"{document}, página {page}"
            else:
                location = str(document)

            output.append(
                f"[{index}] {location}\n{content}"
            )

        else:

            output.append(
                f"[{index}] {_safe_text(chunk)}"
            )

    return "\n\n".join(output)


def _format_citations(
    citations: Sequence[Any],
) -> str:
    """Formata citações."""

    if not citations:
        return "Nenhuma citação disponível."

    output: List[str] = []

    for index, citation in enumerate(
        citations,
        start=1,
    ):

        if isinstance(citation, dict):

            document = (
                citation.get("document")
                or citation.get("document_name")
                or "Documento"
            )

            page = citation.get("page")

            content = (
                citation.get("content")
                or citation.get("excerpt")
                or citation.get("text")
                or ""
            )

            if page is not None:
                location = (
                    f"{document}, página {page}"
                )
            else:
                location = str(document)

            output.append(
                f"[{index}] {location}\n{content}"
            )

        else:

            output.append(
                f"[{index}] {_safe_text(citation)}"
            )

    return "\n\n".join(output)


# ============================================================
# ORCHESTRATE
# ============================================================

def orchestrate(
    question: str,
    mode: str = "auto",
    chunks: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
    top_k: int = 5,
    run_risk: bool = True,
    run_summary: bool = False,
) -> Dict[str, Any]:
    """
    Orquestra todos os agentes.

    Uso:

        result = orchestrate(
            question,
            mode="auto"
        )

    Retorno:

        {
            "success": True,
            "answer": "...",
            "chunks": [...],
            "citations": [...],
            "risk": {...},
            "summary": {...},
            "guard": {...},
            "evaluation": {...}
        }
    """

    question = _safe_text(question)

    if not question:

        return {
            "success": False,
            "answer": "",
            "error": "Pergunta vazia.",
            "chunks": [],
            "citations": [],
            "risk": {},
            "summary": {},
            "guard": {},
            "evaluation": {},
        }

    # ========================================================
    # 1. RAG
    # ========================================================

    if chunks:

        context = list(chunks)
        rag_result = {
            "success": True,
            "chunks": context,
            "citations": list(citations or []),
            "answer": "",
        }

    else:

        rag_result = _run_rag(
            question,
            top_k=top_k,
        )

        context = rag_result.get(
            "chunks",
            [],
        )

        citations = rag_result.get(
            "citations",
            [],
        )

    context = list(context or [])
    citations = list(citations or [])

    # ========================================================
    # 2. RESPOSTA PRINCIPAL
    # ========================================================

    selected_mode = (
        mode or "auto"
    ).lower().strip()

    if selected_mode in (
        "legal",
        "juridico",
        "jurídico",
    ):

        primary = legal_agent(
            question,
            context,
            citations,
        )

    elif selected_mode in (
        "general",
        "geral",
    ):

        primary = general_agent(
            question,
            context,
            citations,
        )

    elif selected_mode in (
        "summary",
        "resumo",
    ):

        primary = summary_agent(
            question,
            context,
        )

    else:

        # ----------------------------------------------------
        # AUTO
        # ----------------------------------------------------

        primary = legal_agent(
            question,
            context,
            citations,
        )

        # Fallback para agente geral.
        if not primary.get("success"):

            logger.warning(
                "Agente Jurídico falhou. "
                "Executando fallback para Agente Geral."
            )

            primary = general_agent(
                question,
                context,
                citations,
            )

    answer = _extract_answer(primary)

    # ========================================================
    # 3. FALLBACK RAG
    # ========================================================

    if not answer and rag_result.get("answer"):

        answer = _safe_text(
            rag_result.get("answer")
        )

    # ========================================================
    # 4. GUARD AGENT
    # ========================================================

    guard = guard_agent(
        question,
        answer,
        context,
    )

    # ========================================================
    # 5. RISCO
    # ========================================================

    risk: Dict[str, Any] = {}

    if run_risk:

        risk = risk_analysis(
            question,
            context,
            citations,
        )

    # ========================================================
    # 6. RESUMO
    # ========================================================

    summary: Dict[str, Any] = {}

    if run_summary:

        summary = summary_agent(
            question,
            context,
        )

    # ========================================================
    # 7. EVALUATION
    # ========================================================

    evaluation: Dict[str, Any] = {}

    if evaluate_answer is not None:

        try:

            evaluation = evaluate_answer(
                question,
                answer,
                context,
                citations,
            )

        except Exception:

            logger.exception(
                "Falha na avaliação da resposta."
            )

            evaluation = {
                "valid": False,
                "overall": 0.0,
                "quality": "Indisponível",
            }

    # ========================================================
    # 8. RESULTADO
    # ========================================================

    success = bool(
        answer
        and guard.get("approved", False)
    )

    return {
        "success": success,

        "answer": answer,

        "agent": primary.get(
            "agent",
            "legal",
        ),

        "mode": selected_mode,

        "chunks": context,

        "reranked": context,

        "citations": citations,

        "risk": risk,

        "risk_analysis": risk,

        "summary": summary,

        "guard": guard,

        "evaluation": evaluation,

        "rag": rag_result,

        "primary": primary,

        "recommendations": evaluation.get(
            "recommendations",
            [],
        ) if isinstance(evaluation, dict) else [],

        "error": (
            primary.get("error")
            if isinstance(primary, dict)
            and not primary.get("success")
            else ""
        ),
    }


# ============================================================
# ALIAS
# ============================================================

def run_orchestrator(
    question: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Alias de compatibilidade.
    """

    return orchestrate(
        question,
        **kwargs,
    )


# ============================================================
# TESTE
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste básico do módulo.
    """

    question = (
        "Qual é o objeto do contrato?"
    )

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "document": "Contrato.pdf",
            "page": 1,
            "content": (
                "O presente contrato tem por objeto "
                "a prestação de serviços de consultoria."
            ),
        }
    ]

    citations = [
        {
            "id": 1,
            "document": "Contrato.pdf",
            "page": 1,
            "chunk_id": "chunk-001",
            "content": chunks[0]["content"],
        }
    ]

    result = orchestrate(
        question=question,
        mode="legal",
        chunks=chunks,
        citations=citations,
        run_risk=False,
        run_summary=False,
    )

    return result


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI ORCHESTRATOR V3.1 - SELF TEST")
    print("=" * 70)

    result = self_test()

    print(
        f"Success: {result.get('success')}"
    )

    print(
        f"Agent: {result.get('agent')}"
    )

    print(
        f"Contextos: "
        f"{len(result.get('chunks', []))}"
    )

    print(
        f"Citações: "
        f"{len(result.get('citations', []))}"
    )

    print(
        f"Resposta: "
        f"{result.get('answer', '')}"
    )

    print(
        f"Evaluation: "
        f"{result.get('evaluation', {})}"
    )

    print("=" * 70)
