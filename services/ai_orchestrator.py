"""
Assistente Jurídico SaaS IA V3.1
services/ai_orchestrator.py

Orquestrador central dos agentes jurídicos.

Agentes:
1. Agente Jurídico
2. Agente de Risco
3. Agente de Resumo
4. Agente Geral
5. RAG / Recuperação
6. Citações
7. Evaluation
8. Guard Agent

Compatibilidade:
- app.py V3
- rag_pipeline.py V3
- chamada orchestrate(query=..., org_id=...)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Sequence


# ============================================================
# LOG
# ============================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# SERVIÇOS
# ============================================================

try:
    from services.ai import generate_answer
except Exception as exc:
    logger.warning("services.ai indisponível: %s", exc)
    generate_answer = None


try:
    from services.rag_pipeline import (
        rag_answer,
        retrieve_and_rerank,
    )
except Exception as exc:
    logger.exception("Falha ao importar rag_pipeline: %s", exc)
    rag_answer = None
    retrieve_and_rerank = None


try:
    from services.evaluation import evaluate_answer
except Exception as exc:
    logger.warning("Evaluation indisponível: %s", exc)
    evaluate_answer = None


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEFAULT_TOP_K = 8
DEFAULT_RERANK_K = 5

MAX_TOP_K = 20
MAX_RERANK_K = 10


# ============================================================
# AGENTES
# ============================================================

AGENT_LEGAL = "legal"
AGENT_RISK = "risk"
AGENT_SUMMARY = "summary"
AGENT_GENERAL = "general"


# ============================================================
# UTILITÁRIOS
# ============================================================

def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _safe_int(
    value: Any,
    default: int,
    minimum: int = 1,
    maximum: int = 100,
) -> int:

    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(value, maximum))


def _safe_list(value: Any) -> List[Any]:

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


def _extract_answer(result: Any) -> str:

    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if not isinstance(result, dict):
        return _safe_text(result)

    for key in (
        "answer",
        "response",
        "text",
        "content",
        "output",
        "message",
        "result",
    ):
        value = result.get(key)

        if value:
            return _safe_text(value)

    return ""


def _extract_chunks(result: Any) -> List[Any]:

    if not isinstance(result, dict):
        return []

    for key in (
        "reranked",
        "chunks",
        "context",
        "results",
        "documents",
        "retrieved",
    ):
        value = result.get(key)

        if value:
            return _safe_list(value)

    return []


def _extract_citations(result: Any) -> List[Any]:

    if not isinstance(result, dict):
        return []

    for key in (
        "citations",
        "references",
        "sources",
        "evidence",
    ):
        value = result.get(key)

        if value:
            return _safe_list(value)

    return []


# ============================================================
# INTENÇÃO
# ============================================================

def detect_intent(query: str) -> str:

    text = _safe_text(query).lower()

    if not text:
        return AGENT_GENERAL

    risk_keywords = [
        "risco",
        "riscos",
        "perigo",
        "ameaça",
        "vulnerabilidade",
        "penalidade",
        "multa",
        "problemática",
        "problemático",
        "ponto crítico",
        "pontos críticos",
        "fragilidade",
        "passivo",
        "ilegal",
        "irregularidade",
    ]

    summary_keywords = [
        "resuma",
        "resumo",
        "resumir",
        "sintetize",
        "síntese",
        "principais pontos",
        "em poucas palavras",
        "resumo do documento",
        "resumo do contrato",
    ]

    legal_keywords = [
        "contrato",
        "contratos",
        "cláusula",
        "clausula",
        "processo",
        "petição",
        "peticao",
        "jurídico",
        "juridico",
        "lei",
        "artigo",
        "jurisprudência",
        "jurisprudencia",
        "autor",
        "réu",
        "reu",
        "obrigação",
        "obrigacao",
        "direito",
        "rescisão",
        "rescisao",
        "indenização",
        "indenizacao",
        "prazo",
        "contestação",
        "contestacao",
        "objeto do contrato",
        "foro",
        "contratante",
        "contratada",
    ]

    if any(x in text for x in risk_keywords):
        return AGENT_RISK

    if any(x in text for x in summary_keywords):
        return AGENT_SUMMARY

    if any(x in text for x in legal_keywords):
        return AGENT_LEGAL

    return AGENT_GENERAL


# ============================================================
# LABEL
# ============================================================

def agent_label(agent: str) -> str:

    labels = {
        AGENT_LEGAL: "Agente Jurídico",
        AGENT_RISK: "Agente de Risco",
        AGENT_SUMMARY: "Agente de Resumo",
        AGENT_GENERAL: "Agente Geral",
    }

    return labels.get(agent, "Agente Geral")


# ============================================================
# INSTRUÇÕES
# ============================================================

def _agent_instruction(agent: str) -> str:

    if agent == AGENT_LEGAL:

        return """
Você é o Agente Jurídico.

Responda utilizando prioritariamente as evidências recuperadas pelo RAG.

REGRAS OBRIGATÓRIAS:

- Não invente fatos.
- Não invente cláusulas.
- Não invente artigos de lei.
- Não invente jurisprudência.
- Não utilize conhecimento externo quando a pergunta depender do documento.
- Se a evidência estiver disponível, responda diretamente.
- Utilize citações [1], [2], [3] quando disponíveis.
- Diferencie informação encontrada no documento de interpretação jurídica.
- Seja objetivo e profissional.
"""

    if agent == AGENT_RISK:

        return """
Você é o Agente de Risco Jurídico.

Analise as evidências recuperadas.

Identifique:

1. Riscos.
2. Pontos de atenção.
3. Inconsistências.
4. Lacunas.
5. Impactos possíveis.
6. Recomendações.

Classifique os riscos como:

- Crítico
- Alto
- Médio
- Baixo

Não invente informações.
Sempre relacione os riscos às evidências.
Use citações [1], [2], [3].
"""

    if agent == AGENT_SUMMARY:

        return """
Você é o Agente de Resumo Jurídico.

Produza um resumo fiel aos documentos.

Estrutura:

### Resumo executivo

### Principais pontos

### Obrigações

### Pontos de atenção

### Evidências

Não invente informações.
Use citações quando disponíveis.
"""

    return """
Você é o Agente Geral do Assistente Jurídico.

Responda de forma clara e objetiva.

Quando a pergunta depender dos documentos:

- utilize as evidências recuperadas;
- não invente informações;
- não invente fatos;
- utilize citações;
- informe quando não houver evidência suficiente.

Nunca apresente uma hipótese como fato.
"""


# ============================================================
# FORMATAÇÃO DE CONTEXTO
# ============================================================

def _format_context(chunks: Sequence[Any]) -> str:

    if not chunks:
        return "Nenhuma evidência foi recuperada."

    output = []

    for index, chunk in enumerate(chunks, 1):

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


# ============================================================
# FORMATAÇÃO DE CITAÇÕES
# ============================================================

def _format_citations(citations: Sequence[Any]) -> str:

    if not citations:
        return "Nenhuma citação disponível."

    output = []

    for index, citation in enumerate(citations, 1):

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
                location = f"{document}, página {page}"
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
# EXECUTAR LLM
# ============================================================

def _execute_llm(
    question: str,
    context: Sequence[Any],
    citations: Sequence[Any],
    agent: str,
) -> Dict[str, Any]:

    if generate_answer is None:

        return {
            "success": False,
            "answer": "",
            "error": "Serviço LLM indisponível.",
        }

    prompt = f"""
{_agent_instruction(agent)}

PERGUNTA DO USUÁRIO:
{question}

EVIDÊNCIAS RECUPERADAS:
{_format_context(context)}

CITAÇÕES:
{_format_citations(citations)}

RESPONDA AGORA:
"""

    try:

        result = generate_answer(prompt)

        answer = _extract_answer(result)

        return {
            "success": bool(answer),
            "answer": answer,
            "agent": agent,
            "raw": result,
        }

    except Exception as exc:

        logger.exception(
            "Falha na execução do LLM para agente %s",
            agent,
        )

        return {
            "success": False,
            "answer": "",
            "agent": agent,
            "error": f"Falha no agente {agent}.",
            "exception": str(exc),
        }


# ============================================================
# RAG
# ============================================================

def _run_rag(
    query: str,
    org_id: int,
    top_k: int,
    rerank_k: int,
    extra_context: str = "",
) -> Dict[str, Any]:

    empty_result = {
        "success": False,
        "retrieved": [],
        "reranked": [],
        "chunks": [],
        "citations": [],
        "answer": "",
        "evidence_count": 0,
    }

    # --------------------------------------------------------
    # MÉTODO PRINCIPAL
    # --------------------------------------------------------

    if rag_answer is not None:

        try:

            result = rag_answer(
                query,
                org_id,
                top_k=top_k,
                rerank_k=rerank_k,
                extra_context=extra_context,
                generate_answer_flag=False,
            )

            if isinstance(result, dict):

                retrieved = _safe_list(
                    result.get("retrieved", [])
                )

                reranked = _safe_list(
                    result.get("reranked", [])
                )

                context = (
                    reranked
                    or _safe_list(result.get("context", []))
                    or _safe_list(result.get("chunks", []))
                    or retrieved
                )

                citations = _extract_citations(result)

                return {
                    **result,
                    "success": bool(context),
                    "retrieved": retrieved,
                    "reranked": reranked,
                    "chunks": context,
                    "citations": citations,
                    "answer": _extract_answer(result),
                    "evidence_count": len(context),
                }

        except TypeError as exc:

            logger.warning(
                "Assinatura antiga do rag_answer detectada: %s",
                exc,
            )

        except Exception as exc:

            logger.exception(
                "Falha no rag_answer: %s",
                exc,
            )

    # --------------------------------------------------------
    # FALLBACK: RETRIEVE + RERANK
    # --------------------------------------------------------

    if retrieve_and_rerank is not None:

        try:

            result = retrieve_and_rerank(
                query,
                org_id,
                top_k=top_k,
                rerank_k=rerank_k,
            )

            if isinstance(result, dict):

                retrieved = _safe_list(
                    result.get("retrieved", [])
                )

                reranked = _safe_list(
                    result.get("reranked", [])
                )

                context = (
                    reranked
                    or _safe_list(result.get("chunks", []))
                    or retrieved
                )

                citations = _extract_citations(result)

                return {
                    **result,
                    "success": bool(context),
                    "retrieved": retrieved,
                    "reranked": reranked,
                    "chunks": context,
                    "citations": citations,
                    "answer": _extract_answer(result),
                    "evidence_count": len(context),
                }

        except Exception as exc:

            logger.exception(
                "Falha no retrieve_and_rerank: %s",
                exc,
            )

    return empty_result


# ============================================================
# GUARD AGENT
# ============================================================

def guard_agent(
    question: str,
    answer: str,
    context: Sequence[Any],
) -> Dict[str, Any]:

    issues = []

    if not question:
        issues.append("Pergunta vazia.")

    if not context:
        issues.append(
            "Nenhuma evidência recuperada."
        )

    if not answer:
        issues.append(
            "Resposta vazia."
        )

    suspicious = [
        "tenho certeza absoluta",
        "garanto que",
        "com certeza absoluta",
        "sem qualquer dúvida",
    ]

    answer_lower = answer.lower()

    for phrase in suspicious:

        if phrase in answer_lower:

            issues.append(
                "Linguagem de certeza excessiva."
            )

            break

    # IMPORTANTE:
    # ausência de contexto não bloqueia tecnicamente
    # a execução do sistema. Ela apenas é sinalizada.

    return {
        "success": len(issues) == 0,
        "approved": True,
        "allowed": True,
        "issues": issues,
        "agent": "guard",
    }


# ============================================================
# ORCHESTRATE
# ============================================================

def orchestrate(
    query: str | None = None,
    org_id: int | None = None,
    mode: str = "auto",
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    extra_context: str = "",
    question: str | None = None,
    chunks: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
    run_risk: bool = False,
    run_summary: bool = False,
) -> Dict[str, Any]:

    started = time.perf_counter()

    # --------------------------------------------------------
    # COMPATIBILIDADE
    # --------------------------------------------------------

    if not query:
        query = question

    query = _safe_text(query)

    # --------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------

    if not query:

        return {
            "success": False,
            "answer": "Digite uma pergunta.",
            "agent": AGENT_GENERAL,
            "agent_label": agent_label(AGENT_GENERAL),
            "intent": AGENT_GENERAL,
            "citations": [],
            "context": [],
            "chunks": [],
            "retrieved": [],
            "reranked": [],
            "evidence_count": 0,
            "guard": {
                "allowed": False,
                "approved": False,
            },
            "evaluation": {},
            "latency_ms": 0,
            "error": "empty_query",
        }

    try:

        org_id = int(org_id)

    except (TypeError, ValueError):

        return {
            "success": False,
            "answer": "Organização inválida.",
            "agent": AGENT_GENERAL,
            "agent_label": agent_label(AGENT_GENERAL),
            "intent": AGENT_GENERAL,
            "citations": [],
            "context": [],
            "chunks": [],
            "retrieved": [],
            "reranked": [],
            "evidence_count": 0,
            "guard": {
                "allowed": False,
                "approved": False,
            },
            "evaluation": {},
            "latency_ms": 0,
            "error": "invalid_org_id",
        }

    # --------------------------------------------------------
    # LIMITES
    # --------------------------------------------------------

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
        MAX_RERANK_K,
    )

    if rerank_k > top_k:
        rerank_k = top_k

    # --------------------------------------------------------
    # AGENTE
    # --------------------------------------------------------

    selected_mode = _safe_text(mode).lower()

    if selected_mode in (
        "",
        "auto",
    ):

        agent = detect_intent(query)

    elif selected_mode in (
        "legal",
        "juridico",
        "jurídico",
    ):

        agent = AGENT_LEGAL

    elif selected_mode in (
        "risk",
        "risco",
    ):

        agent = AGENT_RISK

    elif selected_mode in (
        "summary",
        "resumo",
    ):

        agent = AGENT_SUMMARY

    else:

        agent = AGENT_GENERAL

    logger.info(
        "ORCHESTRATOR | org=%s | agent=%s | query=%s",
        org_id,
        agent,
        query[:120],
    )

    # ========================================================
    # RAG
    # ========================================================

    if chunks:

        context = list(chunks)

        rag_result = {
            "success": True,
            "retrieved": context,
            "reranked": context,
            "chunks": context,
            "citations": list(citations or []),
            "answer": "",
            "evidence_count": len(context),
        }

    else:

        rag_result = _run_rag(
            query=query,
            org_id=org_id,
            top_k=top_k,
            rerank_k=rerank_k,
            extra_context=extra_context,
        )

        context = _safe_list(
            rag_result.get("chunks", [])
        )

        citations = _safe_list(
            rag_result.get("citations", [])
        )

    # ========================================================
    # CONTEXTO EXTRA
    # ========================================================

    if extra_context:

        context = list(context)

        context.append(
            {
                "document": "Contexto fornecido pelo usuário",
                "page": None,
                "content": extra_context,
            }
        )

    # ========================================================
    # AGENTE PRINCIPAL
    # ========================================================

    primary = _execute_llm(
        question=query,
        context=context,
        citations=citations,
        agent=agent,
    )

    answer = _extract_answer(primary)

    # ========================================================
    # FALLBACK
    # ========================================================

    if not answer:

        if rag_result.get("answer"):

            answer = _safe_text(
                rag_result.get("answer")
            )

    # ========================================================
    # GUARD
    # ========================================================

    guard = guard_agent(
        query,
        answer,
        context,
    )

    # ========================================================
    # RISCO OPCIONAL
    # ========================================================

    risk = {}

    if run_risk and context:

        risk = _execute_llm(
            question=(
                "Analise os riscos jurídicos "
                "presentes nas evidências."
            ),
            context=context,
            citations=citations,
            agent=AGENT_RISK,
        )

    # ========================================================
    # RESUMO OPCIONAL
    # ========================================================

    summary = {}

    if run_summary and context:

        summary = _execute_llm(
            question=(
                "Faça um resumo das evidências."
            ),
            context=context,
            citations=citations,
            agent=AGENT_SUMMARY,
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation = {}

    if evaluate_answer is not None and answer:

        try:

            evaluation = evaluate_answer(
                query,
                answer,
                context,
                citations,
            )

        except Exception as exc:

            logger.exception(
                "Falha na avaliação: %s",
                exc,
            )

            evaluation = {
                "valid": False,
                "overall": 0.0,
                "quality": "Indisponível",
            }

    # ========================================================
    # STATUS
    # ========================================================

    latency_ms = int(
        (time.perf_counter() - started) * 1000
    )

    evidence_count = len(context)

    # ========================================================
    # RESPOSTA SEM EVIDÊNCIA
    # ========================================================

    if not context and not answer:

        answer = (
            "Não foi possível gerar uma resposta "
            "com base nas evidências disponíveis."
        )

    # ========================================================
    # RESULTADO
    # ========================================================

    return {
        "success": bool(answer),

        "answer": answer,

        "agent": agent,

        "agent_label": agent_label(agent),

        "intent": (
            "legal_query"
            if agent == AGENT_LEGAL
            else agent
        ),

        "mode": selected_mode,

        "organization_id": org_id,

        "query": query,

        # RAG
        "rag": rag_result,

        "retrieved": rag_result.get(
            "retrieved",
            [],
        ),

        "reranked": rag_result.get(
            "reranked",
            context,
        ),

        "chunks": context,

        "context": context,

        # Evidências
        "citations": citations,

        "evidence_count": evidence_count,

        "evidence_status": (
            "available"
            if evidence_count > 0
            else "none"
        ),

        # Agentes
        "primary": primary,

        "risk": risk,

        "risk_analysis": risk,

        "summary": summary,

        # Segurança
        "guard": guard,

        # Métricas
        "evaluation": evaluation,

        "latency_ms": latency_ms,

        # Compatibilidade
        "recommendations": (
            evaluation.get(
                "recommendations",
                [],
            )
            if isinstance(
                evaluation,
                dict,
            )
            else []
        ),

        "error": (
            primary.get("error", "")
            if isinstance(
                primary,
                dict,
            )
            and not primary.get("success")
            else ""
        ),
    }


# ============================================================
# ANÁLISE JURÍDICA
# ============================================================

def legal_analysis(
    query: str,
    org_id: int,
    **kwargs: Any,
) -> Dict[str, Any]:

    return orchestrate(
        query=query,
        org_id=org_id,
        mode="legal",
        **kwargs,
    )


# ============================================================
# ANÁLISE DE RISCO
# ============================================================

def risk_analysis(
    query: str,
    org_id: int,
    **kwargs: Any,
) -> Dict[str, Any]:

    return orchestrate(
        query=query,
        org_id=org_id,
        mode="risk",
        **kwargs,
    )


# ============================================================
# RESUMO
# ============================================================

def summarize(
    query: str,
    org_id: int,
    **kwargs: Any,
) -> Dict[str, Any]:

    return orchestrate(
        query=query,
        org_id=org_id,
        mode="summary",
        **kwargs,
    )


# ============================================================
# ALIAS
# ============================================================

def run_orchestrator(
    query: str | None = None,
    question: str | None = None,
    org_id: int | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    return orchestrate(
        query=query or question,
        org_id=org_id,
        **kwargs,
    )


# ============================================================
# STATUS DOS 8 AGENTES
# ============================================================

def orchestrator_status() -> Dict[str, Any]:

    return {
        "configured": True,

        "agents": [
            {
                "id": "legal",
                "name": "Agente Jurídico",
                "status": "active",
            },
            {
                "id": "risk",
                "name": "Agente de Risco",
                "status": "active",
            },
            {
                "id": "summary",
                "name": "Agente de Resumo",
                "status": "active",
            },
            {
                "id": "general",
                "name": "Agente Geral",
                "status": "active",
            },
            {
                "id": "rag",
                "name": "RAG / Recuperação",
                "status": "active",
            },
            {
                "id": "citations",
                "name": "Citações e Evidências",
                "status": "active",
            },
            {
                "id": "evaluation",
                "name": "Evaluation / Métricas",
                "status": "active",
            },
            {
                "id": "guard",
                "name": "Guard Agent",
                "status": "active",
            },
        ],

        "default_top_k": DEFAULT_TOP_K,

        "default_rerank_k": DEFAULT_RERANK_K,

        "max_top_k": MAX_TOP_K,

        "max_rerank_k": MAX_RERANK_K,
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:

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

    return orchestrate(
        query="Qual é o objeto do contrato?",
        org_id=1,
        mode="legal",
        chunks=chunks,
        citations=citations,
        run_risk=False,
        run_summary=False,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI ORCHESTRATOR V3.1")
    print("=" * 70)

    result = self_test()

    print("Success:", result.get("success"))
    print("Agent:", result.get("agent"))
    print("Intent:", result.get("intent"))
    print("Evidências:", result.get("evidence_count"))
    print("Citações:", len(result.get("citations", [])))
    print("Guard:", result.get("guard"))
    print("Evaluation:", result.get("evaluation"))
    print("Resposta:", result.get("answer"))

    print("=" * 70)
