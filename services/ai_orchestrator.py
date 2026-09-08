"""
Assistente Jurídico SaaS IA V3.2
services/ai_orchestrator.py

Orquestrador central dos agentes jurídicos.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Sequence


logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


try:
    from services.ai import (
        generate_answer,
        generate_answer_result,
        ai_status,
    )
except Exception as exc:
    logger.warning("services.ai indisponível: %s", exc)
    generate_answer = None
    generate_answer_result = None
    ai_status = None


try:
    from services.rag_pipeline import (
        rag_answer,
        retrieve_and_rerank,
    )
except Exception as exc:
    logger.warning("services.rag_pipeline indisponível: %s", exc)
    rag_answer = None
    retrieve_and_rerank = None


try:
    from services.evaluation import evaluate_answer
except Exception as exc:
    logger.warning("Evaluation indisponível: %s", exc)
    evaluate_answer = None


DEFAULT_TOP_K = 8
DEFAULT_RERANK_K = 5
MAX_TOP_K = 20
MAX_RERANK_K = 10

AGENT_LEGAL = "legal"
AGENT_RISK = "risk"
AGENT_SUMMARY = "summary"
AGENT_GENERAL = "general"
AGENT_RAG = "rag"
AGENT_CITATIONS = "citations"
AGENT_EVALUATION = "evaluation"
AGENT_GUARD = "guard"


# ============================================================
# GUARDRAILS DE QUALIDADE (NOVO — item 2.3)
# ============================================================
#
# Limiares que definem quando uma resposta é considerada
# confiável o suficiente para ser apresentada sem alerta.
#
# Calibrar conforme uso real; começam conservadores.

MIN_OVERALL_SCORE_APPROVED = 0.55
MIN_CITATION_COVERAGE_APPROVED = 0.99  # qualquer citação inválida reprova


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_int(value: Any, default: int, minimum: int = 1, maximum: int = 100) -> int:
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
    for key in ("answer", "response", "text", "content", "output", "message", "result"):
        value = result.get(key)
        if value:
            return _safe_text(value)
    return ""


def _extract_chunks(result: Any) -> List[Any]:
    if not isinstance(result, dict):
        return []
    for key in ("reranked", "chunks", "context", "results", "documents", "retrieved"):
        value = result.get(key)
        if value:
            return _safe_list(value)
    return []


def _extract_citations(result: Any) -> List[Any]:
    if not isinstance(result, dict):
        return []
    for key in ("citations", "references", "sources", "evidence"):
        value = result.get(key)
        if value:
            return _safe_list(value)
    return []


def detect_intent(query: str) -> str:
    text = _safe_text(query).lower()
    if not text:
        return AGENT_GENERAL

    risk_keywords = [
        "risco", "riscos", "perigo", "ameaça", "vulnerabilidade", "penalidade",
        "multa", "problemática", "problemático", "ponto crítico", "pontos críticos",
        "fragilidade", "passivo", "ilegal", "irregularidade", "inconsistência",
        "inconsistências", "lacuna", "lacunas",
    ]
    summary_keywords = [
        "resuma", "resumo", "resumir", "sintetize", "síntese", "principais pontos",
        "em poucas palavras", "resumo do documento", "resumo do contrato",
        "faça um resumo",
    ]
    legal_keywords = [
        "contrato", "contratos", "cláusula", "clausula", "processo", "petição",
        "peticao", "jurídico", "juridico", "lei", "artigo", "jurisprudência",
        "jurisprudencia", "autor", "réu", "reu", "obrigação", "obrigacao",
        "direito", "rescisão", "rescisao", "indenização", "indenizacao", "prazo",
        "contestação", "contestacao", "objeto do contrato", "foro", "contratante",
        "contratada", "responsabilidade", "responsabilidades", "penhora",
        "sentença", "sentenca", "recurso", "peticionamento",
    ]

    if any(k in text for k in risk_keywords):
        return AGENT_RISK
    if any(k in text for k in summary_keywords):
        return AGENT_SUMMARY
    if any(k in text for k in legal_keywords):
        return AGENT_LEGAL
    return AGENT_GENERAL


def agent_label(agent: str) -> str:
    labels = {
        AGENT_LEGAL: "Agente Jurídico",
        AGENT_RISK: "Agente de Risco",
        AGENT_SUMMARY: "Agente de Resumo",
        AGENT_GENERAL: "Agente Geral",
        AGENT_RAG: "RAG / Recuperação",
        AGENT_CITATIONS: "Citações e Evidências",
        AGENT_EVALUATION: "Evaluation / Métricas",
        AGENT_GUARD: "Guard Agent",
    }
    return labels.get(agent, "Agente Geral")


def _agent_instruction(agent: str) -> str:
    if agent == AGENT_LEGAL:
        return """
Você é o Agente Jurídico do Assistente Jurídico SaaS IA.

MISSÃO:
Analisar documentos jurídicos e responder perguntas com base
nas evidências recuperadas pelo RAG.

REGRAS:
- Não invente fatos, cláusulas, artigos de lei, jurisprudência,
  números de processos, datas ou valores.
- Priorize as evidências recuperadas.
- Diferencie fato de interpretação.
- Informe quando a evidência for insuficiente.
- Utilize [1], [2], [3] conforme as fontes disponíveis.
- Seja objetivo, profissional e auditável.

Se não houver evidência suficiente, diga explicitamente:
"Não há evidência suficiente nos documentos disponibilizados."
"""
    if agent == AGENT_RISK:
        return """
Você é o Agente de Risco Jurídico.

MISSÃO:
Identificar riscos e pontos críticos existentes nas evidências
documentais fornecidas, classificando-os como Crítico/Alto/Médio/Baixo.

REGRAS:
- Todo risco deve estar relacionado a uma evidência.
- Não invente fatos, cláusulas ou legislação.
- Utilize citações [1], [2], [3].
- Se não houver evidência suficiente, informe isso.
"""
    if agent == AGENT_SUMMARY:
        return """
Você é o Agente de Resumo Jurídico.

MISSÃO:
Produzir um resumo fiel, objetivo e estruturado dos documentos
fornecidos como evidência.

REGRAS:
- Não invente informações nem extrapole o conteúdo documental.
- Preserve datas e valores exatamente quando disponíveis.
- Utilize citações [1], [2], [3].
"""
    return """
Você é o Agente Geral do Assistente Jurídico SaaS IA.

MISSÃO:
Responder perguntas gerais relacionadas ao conteúdo
disponibilizado pelo usuário.

REGRAS:
- Utilize as evidências quando a pergunta depender delas.
- Não invente informações nem referências.
- Informe quando não houver evidência suficiente.
- Nunca apresente hipótese como fato.
"""


def _execute_llm(question: str, context: Sequence[Any], citations: Sequence[Any], agent: str) -> Dict[str, Any]:
    started = time.perf_counter()

    if generate_answer is None:
        return {
            "success": False, "answer": "", "agent": agent,
            "agent_label": agent_label(agent),
            "error": "Serviço LLM indisponível.", "latency_ms": 0,
        }

    instruction = _agent_instruction(agent)

    try:
        result = generate_answer(query=question, chunks=list(context), agent_instruction=instruction)
        answer = _extract_answer(result)
        latency_ms = int((time.perf_counter() - started) * 1000)

        if not answer:
            return {
                "success": False, "answer": "", "agent": agent,
                "agent_label": agent_label(agent),
                "error": "O LLM não retornou conteúdo.",
                "latency_ms": latency_ms, "raw": result,
            }

        return {
            "success": True, "answer": answer, "agent": agent,
            "agent_label": agent_label(agent), "latency_ms": latency_ms,
            "error": None, "raw": result,
        }

    except TypeError as exc:
        logger.warning("Compatibilidade AI Service: %s", exc)
        try:
            result = generate_answer(question, list(context))
            answer = _extract_answer(result)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return {
                "success": bool(answer), "answer": answer, "agent": agent,
                "agent_label": agent_label(agent), "latency_ms": latency_ms,
                "error": None if answer else "LLM sem resposta.", "raw": result,
            }
        except Exception as fallback_exc:
            logger.exception("Falha no fallback do AI Service.")
            latency_ms = int((time.perf_counter() - started) * 1000)
            return {
                "success": False, "answer": "", "agent": agent,
                "agent_label": agent_label(agent), "latency_ms": latency_ms,
                "error": f"Falha no agente {agent_label(agent)}.",
                "exception": str(fallback_exc),
            }

    except Exception as exc:
        logger.exception("Falha na execução do agente %s", agent)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "success": False, "answer": "", "agent": agent,
            "agent_label": agent_label(agent), "latency_ms": latency_ms,
            "error": f"Falha no agente {agent_label(agent)}.", "exception": str(exc),
        }


def _run_rag(query: str, org_id: int, top_k: int, rerank_k: int, extra_context: str = "") -> Dict[str, Any]:
    empty_result = {
        "success": False, "retrieved": [], "reranked": [], "chunks": [], "context": [],
        "citations": [], "answer": "", "evidence_count": 0,
        "evidence_status": "insufficient", "error": None,
    }

    if callable(rag_answer):
        try:
            result = rag_answer(
                query=query, org_id=org_id, top_k=top_k, rerank_k=rerank_k,
                extra_context=extra_context, generate_answer_flag=False,
            )
            if isinstance(result, dict):
                retrieved = _safe_list(result.get("retrieved", []))
                reranked = _safe_list(result.get("reranked", []))
                context = (
                    _safe_list(result.get("context", []))
                    or reranked
                    or _safe_list(result.get("chunks", []))
                    or retrieved
                )
                citations = _extract_citations(result)
                return {
                    **result, "success": bool(context), "retrieved": retrieved,
                    "reranked": reranked, "chunks": context, "context": context,
                    "citations": citations, "answer": _extract_answer(result),
                    "evidence_count": len(context),
                    "evidence_status": "available" if context else "insufficient",
                }
        except TypeError as exc:
            logger.warning("Assinatura antiga do rag_answer: %s", exc)
        except Exception as exc:
            logger.exception("Falha no rag_answer: %s", exc)

    if callable(retrieve_and_rerank):
        try:
            result = retrieve_and_rerank(query=query, org_id=org_id, top_k=top_k, rerank_k=rerank_k)
            if isinstance(result, dict):
                retrieved = _safe_list(result.get("retrieved", []))
                reranked = _safe_list(result.get("reranked", []))
                context = reranked or _safe_list(result.get("chunks", [])) or retrieved
                citations = _extract_citations(result)
                return {
                    **result, "success": bool(context), "retrieved": retrieved,
                    "reranked": reranked, "chunks": context, "context": context,
                    "citations": citations, "answer": "",
                    "evidence_count": len(context),
                    "evidence_status": "available" if context else "insufficient",
                }
        except Exception as exc:
            logger.exception("Falha no retrieve_and_rerank: %s", exc)

    return empty_result


# ============================================================
# GUARD AGENT — REESCRITO (item 2.2 + 2.3)
# ============================================================
#
# Antes: sempre retornava approved=True/allowed=True,
# independente dos "issues" encontrados, e não usava
# nenhuma métrica objetiva de qualidade.
#
# Agora: usa o resultado real de evaluate_answer() (quando
# disponível) para decidir se a resposta é aprovada,
# aprovada-com-ressalvas, ou reprovada. A citação inválida
# (cite [N] que não existe nas evidências) passa a reprovar
# a resposta de verdade, não só ser "detectada e ignorada".

def guard_agent(
    question: str,
    answer: str,
    context: Sequence[Any],
    evaluation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Verificação de segurança e qualidade da resposta.

    Combina checagens estruturais básicas (pergunta/resposta/
    evidência vazias, linguagem de certeza excessiva) com as
    métricas objetivas calculadas por services.evaluation
    (quando disponíveis): citation_coverage e overall score.
    """

    issues: List[str] = []

    question = _safe_text(question)
    answer = _safe_text(answer)

    if not question:
        issues.append("Pergunta vazia.")

    if not context:
        issues.append("Nenhuma evidência recuperada.")

    if not answer:
        issues.append("Resposta vazia.")

    suspicious = [
        "tenho certeza absoluta", "garanto que", "com certeza absoluta",
        "sem qualquer dúvida", "100% certo",
    ]
    answer_lower = answer.lower()
    for phrase in suspicious:
        if phrase in answer_lower:
            issues.append("Linguagem de certeza excessiva.")
            break

    # ------------------------------------------------------
    # Avaliação objetiva (NOVO)
    # ------------------------------------------------------
    overall = None
    citation_coverage_score = None

    if isinstance(evaluation, dict) and evaluation:
        overall = evaluation.get("overall")
        citation_coverage_score = evaluation.get("citation_coverage")

        if citation_coverage_score is not None and citation_coverage_score < MIN_CITATION_COVERAGE_APPROVED:
            issues.append(
                "Citação inválida detectada: a resposta referencia "
                "uma fonte [N] que não corresponde às evidências fornecidas."
            )

        if overall is not None and overall < MIN_OVERALL_SCORE_APPROVED:
            issues.append(
                f"Score de qualidade abaixo do limiar aceitável "
                f"({overall:.2f} < {MIN_OVERALL_SCORE_APPROVED})."
            )

    # ------------------------------------------------------
    # Decisão final — agora reflete os issues de verdade
    # ------------------------------------------------------
    approved = len(issues) == 0

    return {
        "success": approved,
        "approved": approved,
        "allowed": True,  # nunca bloqueamos o envio da resposta ao usuário,
                            # mas approved=False deve disparar um aviso visível na UI
        "issues": issues,
        "issue_count": len(issues),
        "overall_score": overall,
        "citation_coverage": citation_coverage_score,
        "agent": AGENT_GUARD,
        "agent_label": agent_label(AGENT_GUARD),
    }


def _run_risk_agent(query: str, context: Sequence[Any], citations: Sequence[Any]) -> Dict[str, Any]:
    return _execute_llm(
        question=f"Analise os riscos jurídicos relacionados à seguinte solicitação:\n\n{query}",
        context=context, citations=citations, agent=AGENT_RISK,
    )


def _run_summary_agent(query: str, context: Sequence[Any], citations: Sequence[Any]) -> Dict[str, Any]:
    return _execute_llm(
        question=f"Produza um resumo jurídico das evidências relacionadas à seguinte solicitação:\n\n{query}",
        context=context, citations=citations, agent=AGENT_SUMMARY,
    )


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

    if not query:
        query = question
    query = _safe_text(query)

    if not query:
        return {
            "success": False, "answer": "Digite uma pergunta.", "agent": AGENT_GENERAL,
            "agent_label": agent_label(AGENT_GENERAL), "intent": AGENT_GENERAL,
            "citations": [], "context": [], "chunks": [], "retrieved": [], "reranked": [],
            "evidence_count": 0, "guard": {"allowed": False, "approved": False},
            "evaluation": {}, "latency_ms": 0, "error": "empty_query",
        }

    try:
        org_id = int(org_id)
    except (TypeError, ValueError):
        return {
            "success": False, "answer": "Organização inválida.", "agent": AGENT_GENERAL,
            "agent_label": agent_label(AGENT_GENERAL), "intent": AGENT_GENERAL,
            "citations": [], "context": [], "chunks": [], "retrieved": [], "reranked": [],
            "evidence_count": 0, "guard": {"allowed": False, "approved": False},
            "evaluation": {}, "latency_ms": 0, "error": "invalid_org_id",
        }

    top_k = _safe_int(top_k, DEFAULT_TOP_K, 1, MAX_TOP_K)
    rerank_k = _safe_int(rerank_k, DEFAULT_RERANK_K, 1, MAX_RERANK_K)
    if rerank_k > top_k:
        rerank_k = top_k

    selected_mode = _safe_text(mode).lower()

    # ------------------------------------------------------
    # CORREÇÃO (item 2.5): mode explícito da UI tem prioridade
    # sobre detecção automática por keyword.
    # ------------------------------------------------------
    if selected_mode in ("", "auto"):
        agent = detect_intent(query)
    elif selected_mode in ("legal", "juridico", "jurídico"):
        agent = AGENT_LEGAL
    elif selected_mode in ("risk", "risco"):
        agent = AGENT_RISK
    elif selected_mode in ("summary", "resumo"):
        agent = AGENT_SUMMARY
    elif selected_mode in ("general", "geral"):
        agent = AGENT_GENERAL
    else:
        agent = AGENT_GENERAL

    logger.info("ORCHESTRATOR | org=%s | agent=%s | query=%s", org_id, agent, query[:120])

    if chunks:
        context = list(chunks)
        rag_result = {
            "success": True, "retrieved": context, "reranked": context,
            "chunks": context, "context": context, "citations": list(citations or []),
            "answer": "", "evidence_count": len(context), "evidence_status": "available",
        }
    else:
        rag_result = _run_rag(query=query, org_id=org_id, top_k=top_k, rerank_k=rerank_k, extra_context=extra_context)
        context = _safe_list(rag_result.get("context", []))
        if not context:
            context = _safe_list(rag_result.get("chunks", []))
        citations = _safe_list(rag_result.get("citations", []))

    if extra_context:
        already_added = any(
            isinstance(item, dict) and item.get("chunk_id") == "user_input"
            for item in context
        )
        if not already_added:
            context = list(context)
            context.append({
                "chunk_id": "user_input",
                "document": "Contexto fornecido pelo usuário",
                "document_id": None, "page": "N/D", "content": extra_context,
                "reranker_score": 1.0, "retriever_score": 1.0,
            })

    primary = _execute_llm(question=query, context=context, citations=citations, agent=agent)
    answer = _extract_answer(primary)

    if not answer:
        rag_answer_text = _extract_answer(rag_result)
        if rag_answer_text:
            answer = rag_answer_text

    # ------------------------------------------------------
    # CORREÇÃO (item 2.3): calcular a avaliação ANTES do guard,
    # para que o guard possa usar métricas objetivas em vez de
    # só um checklist superficial.
    # ------------------------------------------------------
    evaluation: Dict[str, Any] = {}
    if callable(evaluate_answer) and answer:
        try:
            evaluation = evaluate_answer(query, answer, context, citations)
            if not isinstance(evaluation, dict):
                evaluation = {"valid": True, "overall": 0.0, "quality": "Disponível"}
        except Exception as exc:
            logger.exception("Falha na avaliação.")
            evaluation = {
                "valid": False, "overall": 0.0, "quality": "Indisponível",
                "error": str(exc)[:300],
            }

    guard = guard_agent(query, answer, context, evaluation=evaluation)

    risk: Dict[str, Any] = {}
    if run_risk:
        if context:
            risk = _run_risk_agent(query=query, context=context, citations=citations)
        else:
            risk = {
                "success": False, "answer": "", "agent": AGENT_RISK,
                "agent_label": agent_label(AGENT_RISK),
                "error": "Não há evidências para análise de risco.", "latency_ms": 0,
            }

    summary: Dict[str, Any] = {}
    if run_summary:
        if context:
            summary = _run_summary_agent(query=query, context=context, citations=citations)
        else:
            summary = {
                "success": False, "answer": "", "agent": AGENT_SUMMARY,
                "agent_label": agent_label(AGENT_SUMMARY),
                "error": "Não há evidências para gerar resumo.", "latency_ms": 0,
            }

    latency_ms = int((time.perf_counter() - started) * 1000)
    evidence_count = len(context)

    if not context and not answer:
        answer = "Não foi possível gerar uma resposta com base nas evidências disponíveis."

    primary_status = "operational" if primary.get("success") else "error"

    return {
        "success": bool(answer),
        "answer": answer,
        "agent": agent,
        "agent_label": agent_label(agent),
        "agent_status": primary_status,
        "intent": "legal_query" if agent == AGENT_LEGAL else agent,
        "mode": selected_mode,
        "organization_id": org_id,
        "query": query,
        "rag": rag_result,
        "retrieved": rag_result.get("retrieved", []),
        "reranked": rag_result.get("reranked", context),
        "chunks": context,
        "context": context,
        "citations": citations,
        "evidence_count": evidence_count,
        "evidence_status": "available" if evidence_count > 0 else "none",
        "primary": primary,
        "risk": risk,
        "risk_analysis": risk,
        "summary": summary,
        "guard": guard,
        "evaluation": evaluation,
        "latency_ms": latency_ms,
        "recommendations": (
            evaluation.get("recommendations", []) if isinstance(evaluation, dict) else []
        ),
        "error": (
            primary.get("error", "") if isinstance(primary, dict) and not primary.get("success") else ""
        ),
    }


def legal_analysis(query: str, org_id: int, **kwargs: Any) -> Dict[str, Any]:
    return orchestrate(query=query, org_id=org_id, mode="legal", **kwargs)


def risk_analysis(query: str, org_id: int, **kwargs: Any) -> Dict[str, Any]:
    return orchestrate(query=query, org_id=org_id, mode="risk", **kwargs)


def summarize(query: str, org_id: int, **kwargs: Any) -> Dict[str, Any]:
    return orchestrate(query=query, org_id=org_id, mode="summary", **kwargs)


def run_orchestrator(query: str | None = None, question: str | None = None, org_id: int | None = None, **kwargs: Any) -> Dict[str, Any]:
    return orchestrate(query=query or question, org_id=org_id, **kwargs)


def orchestrator_status() -> Dict[str, Any]:
    ai_info: Dict[str, Any] = {}
    if callable(ai_status):
        try:
            result = ai_status()
            if isinstance(result, dict):
                ai_info = result
        except Exception as exc:
            ai_info = {"configured": False, "status": "error", "error": str(exc)[:300]}

    ai_configured = bool(ai_info.get("configured", False))
    ai_status_value = ai_info.get("status", "not_configured")

    rag_active = callable(rag_answer) or callable(retrieve_and_rerank)
    evaluation_active = callable(evaluate_answer)

    components = [
        {"id": AGENT_LEGAL, "name": "Agente Jurídico", "type": "agent", "status": "operational" if callable(generate_answer) else "error"},
        {"id": AGENT_RISK, "name": "Agente de Risco", "type": "agent", "status": "operational" if callable(generate_answer) else "error"},
        {"id": AGENT_SUMMARY, "name": "Agente de Resumo", "type": "agent", "status": "operational" if callable(generate_answer) else "error"},
        {"id": AGENT_GENERAL, "name": "Agente Geral", "type": "agent", "status": "operational" if callable(generate_answer) else "error"},
        {"id": AGENT_RAG, "name": "RAG / Recuperação", "type": "pipeline", "status": "operational" if rag_active else "error"},
        {"id": AGENT_CITATIONS, "name": "Citações e Evidências", "type": "pipeline", "status": "operational" if rag_active else "degraded"},
        {"id": AGENT_EVALUATION, "name": "Evaluation / Métricas", "type": "pipeline", "status": "operational" if evaluation_active else "unavailable"},
        {"id": AGENT_GUARD, "name": "Guard Agent", "type": "security", "status": "operational"},
    ]

    operational = sum(1 for c in components if c["status"] == "operational")
    total = len(components)

    return {
        "configured": bool(callable(generate_answer)),
        "overall_status": "operational" if operational == total else ("degraded" if operational > 0 else "error"),
        "operational_count": operational,
        "total_components": total,
        "agents": components,
        "components": components,
        "ai_service": {
            "available": callable(generate_answer), "configured": ai_configured,
            "status": ai_status_value, "provider": ai_info.get("provider"),
            "model": ai_info.get("model"), "temperature": ai_info.get("temperature"),
            "max_tokens": ai_info.get("max_tokens"),
        },
        "rag": {"available": rag_active, "status": "operational" if rag_active else "error"},
        "evaluation": {"available": evaluation_active, "status": "operational" if evaluation_active else "unavailable"},
        "guard": {"available": True, "status": "operational"},
        "default_top_k": DEFAULT_TOP_K,
        "default_rerank_k": DEFAULT_RERANK_K,
        "max_top_k": MAX_TOP_K,
        "max_rerank_k": MAX_RERANK_K,
    }


def orchestrator_health() -> Dict[str, Any]:
    status = orchestrator_status()
    return {
        "healthy": status.get("overall_status") == "operational",
        "status": status.get("overall_status", "error"),
        "operational": status.get("operational_count", 0),
        "total": status.get("total_components", 0),
        "ai_service": status.get("ai_service", {}),
        "rag": status.get("rag", {}),
        "evaluation": status.get("evaluation", {}),
        "guard": status.get("guard", {}),
    }
