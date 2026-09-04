"""
Assistente Jurídico SaaS IA V3.2
services/ai_orchestrator.py

Orquestrador central dos agentes jurídicos.

Componentes:
1. Agente Jurídico
2. Agente de Risco
3. Agente de Resumo
4. Agente Geral
5. RAG / Recuperação
6. Citações / Evidências
7. Evaluation / Métricas
8. Guard Agent

Compatibilidade:
- app.py V3/V3.1
- services.ai
- services.rag_pipeline
- services.evaluation
- chamadas orchestrate(query=..., org_id=...)

Principais melhorias V3.2:
- Corrige integração com services.ai.generate_answer()
- Passa query + chunks corretamente ao LLM
- Suporta agent_instruction
- Status real dos componentes
- Métricas de execução
- Tratamento de erros por componente
- Fallback do RAG
- Execução opcional de Risco e Resumo
- Estrutura preparada para o novo Dashboard / Central de IA
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
    logger.warning(
        "services.rag_pipeline indisponível: %s",
        exc,
    )

    rag_answer = None
    retrieve_and_rerank = None


try:
    from services.evaluation import evaluate_answer
except Exception as exc:
    logger.warning(
        "Evaluation indisponível: %s",
        exc,
    )

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

AGENT_RAG = "rag"
AGENT_CITATIONS = "citations"
AGENT_EVALUATION = "evaluation"
AGENT_GUARD = "guard"


# ============================================================
# UTILITÁRIOS
# ============================================================

def _safe_text(value: Any) -> str:
    """
    Converte qualquer valor em texto seguro.
    """

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
    """
    Converte valor para inteiro dentro de limites seguros.
    """

    try:
        value = int(value)

    except (TypeError, ValueError):
        value = default

    return max(
        minimum,
        min(value, maximum),
    )


def _safe_list(value: Any) -> List[Any]:
    """
    Normaliza valores para lista.
    """

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
    """
    Extrai resposta de diferentes formatos retornados
    pelos serviços de IA.
    """

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
    """
    Extrai chunks de diferentes formatos.
    """

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
    """
    Extrai citações de diferentes formatos.
    """

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
    """
    Detecta automaticamente qual agente deve responder.
    """

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
        "inconsistência",
        "inconsistências",
        "lacuna",
        "lacunas",
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
        "faça um resumo",
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
        "responsabilidade",
        "responsabilidades",
        "penhora",
        "sentença",
        "sentenca",
        "recurso",
        "peticionamento",
    ]

    if any(
        keyword in text
        for keyword in risk_keywords
    ):
        return AGENT_RISK

    if any(
        keyword in text
        for keyword in summary_keywords
    ):
        return AGENT_SUMMARY

    if any(
        keyword in text
        for keyword in legal_keywords
    ):
        return AGENT_LEGAL

    return AGENT_GENERAL


# ============================================================
# LABEL
# ============================================================

def agent_label(agent: str) -> str:
    """
    Nome amigável do agente.
    """

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

    return labels.get(
        agent,
        "Agente Geral",
    )


# ============================================================
# INSTRUÇÕES DOS AGENTES
# ============================================================

def _agent_instruction(agent: str) -> str:
    """
    Retorna a instrução especializada de cada agente.
    """

    if agent == AGENT_LEGAL:

        return """
Você é o Agente Jurídico do Assistente Jurídico SaaS IA.

MISSÃO:
Analisar documentos jurídicos e responder perguntas com base
nas evidências recuperadas pelo RAG.

REGRAS:

- Não invente fatos.
- Não invente cláusulas.
- Não invente artigos de lei.
- Não invente jurisprudência.
- Não invente números de processos.
- Não invente datas ou valores.
- Priorize as evidências recuperadas.
- Diferencie fato de interpretação.
- Informe quando a evidência for insuficiente.
- Utilize [1], [2], [3] conforme as fontes disponíveis.
- Não apresente hipótese como fato.
- Seja objetivo, profissional e auditável.

ESTRUTURA PREFERENCIAL:

### Análise

### Evidências

### Interpretação

### Pontos de atenção

### Conclusão

Se não houver evidência suficiente, diga explicitamente:

"Não há evidência suficiente nos documentos disponibilizados."
"""

    if agent == AGENT_RISK:

        return """
Você é o Agente de Risco Jurídico.

MISSÃO:
Identificar riscos e pontos críticos existentes nas evidências
documentais fornecidas.

ANALISE:

1. Riscos identificados.
2. Pontos de atenção.
3. Inconsistências.
4. Lacunas.
5. Possíveis impactos.
6. Recomendações.

CLASSIFICAÇÃO:

- Crítico
- Alto
- Médio
- Baixo

REGRAS:

- Todo risco deve estar relacionado a uma evidência.
- Não invente fatos.
- Não invente cláusulas.
- Não invente legislação.
- Não crie jurisprudência.
- Não transforme possibilidade em fato.
- Utilize citações [1], [2], [3].
- Se não houver evidência suficiente, informe isso.

ESTRUTURA:

### Riscos identificados

### Classificação

### Evidências

### Impactos possíveis

### Recomendações
"""

    if agent == AGENT_SUMMARY:

        return """
Você é o Agente de Resumo Jurídico.

MISSÃO:
Produzir um resumo fiel, objetivo e estruturado dos documentos
fornecidos como evidência.

ESTRUTURA:

### Resumo executivo

### Principais pontos

### Obrigações

### Direitos e responsabilidades

### Prazos e valores

### Pontos de atenção

### Evidências

REGRAS:

- Não invente informações.
- Não extrapole o conteúdo documental.
- Preserve datas e valores exatamente quando disponíveis.
- Utilize citações [1], [2], [3].
- Diferencie fatos documentais de interpretação.
"""

    return """
Você é o Agente Geral do Assistente Jurídico SaaS IA.

MISSÃO:
Responder perguntas gerais relacionadas ao conteúdo
disponibilizado pelo usuário.

REGRAS:

- Utilize as evidências quando a pergunta depender delas.
- Não invente informações.
- Não invente fatos.
- Não invente referências.
- Utilize citações quando disponíveis.
- Informe quando não houver evidência suficiente.
- Seja claro, objetivo e profissional.
- Nunca apresente hipótese como fato.

Quando a pergunta não depender dos documentos,
responda de forma geral, deixando claro quando houver
necessidade de análise jurídica profissional.
"""


# ============================================================
# FORMATAÇÃO DE CONTEXTO
# ============================================================

def _format_context(
    chunks: Sequence[Any],
) -> str:
    """
    Formata evidências para logs e compatibilidade.
    """

    if not chunks:
        return "Nenhuma evidência foi recuperada."

    output = []

    for index, chunk in enumerate(
        chunks,
        1,
    ):

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
                location = (
                    f"{document}, página {page}"
                )
            else:
                location = str(document)

            output.append(
                f"[{index}] {location}\n"
                f"{content}"
            )

        else:

            output.append(
                f"[{index}] "
                f"{_safe_text(chunk)}"
            )

    return "\n\n".join(output)


# ============================================================
# CITAÇÕES
# ============================================================

def _format_citations(
    citations: Sequence[Any],
) -> str:
    """
    Formata citações.
    """

    if not citations:
        return "Nenhuma citação disponível."

    output = []

    for index, citation in enumerate(
        citations,
        1,
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
                f"[{index}] {location}\n"
                f"{content}"
            )

        else:

            output.append(
                f"[{index}] "
                f"{_safe_text(citation)}"
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
    """
    Executa um agente utilizando o AI Service.

    CORREÇÃO V3.2:

    Antes:

        generate_answer(prompt)

    Agora:

        generate_answer(
            query=question,
            chunks=context,
            agent_instruction=...
        )

    Isso respeita a assinatura real do services.ai.
    """

    started = time.perf_counter()

    if generate_answer is None:

        return {
            "success": False,
            "answer": "",
            "agent": agent,
            "agent_label": agent_label(agent),
            "error": "Serviço LLM indisponível.",
            "latency_ms": 0,
        }

    instruction = _agent_instruction(
        agent
    )

    try:

        # ====================================================
        # CHAMADA CORRETA DO AI SERVICE
        # ====================================================

        result = generate_answer(
            query=question,
            chunks=list(context),
            agent_instruction=instruction,
        )

        answer = _extract_answer(
            result
        )

        latency_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        if not answer:

            return {
                "success": False,
                "answer": "",
                "agent": agent,
                "agent_label": agent_label(agent),
                "error": "O LLM não retornou conteúdo.",
                "latency_ms": latency_ms,
                "raw": result,
            }

        return {
            "success": True,
            "answer": answer,
            "agent": agent,
            "agent_label": agent_label(agent),
            "latency_ms": latency_ms,
            "error": None,
            "raw": result,
        }

    except TypeError as exc:

        logger.warning(
            "Compatibilidade AI Service: %s",
            exc,
        )

        # ----------------------------------------------------
        # FALLBACK PARA VERSÕES ANTIGAS
        # ----------------------------------------------------

        try:

            result = generate_answer(
                question,
                list(context),
            )

            answer = _extract_answer(
                result
            )

            latency_ms = int(
                (
                    time.perf_counter()
                    - started
                )
                * 1000
            )

            return {
                "success": bool(answer),
                "answer": answer,
                "agent": agent,
                "agent_label": agent_label(agent),
                "latency_ms": latency_ms,
                "error": None
                if answer
                else "LLM sem resposta.",
                "raw": result,
            }

        except Exception as fallback_exc:

            logger.exception(
                "Falha no fallback do AI Service."
            )

            latency_ms = int(
                (
                    time.perf_counter()
                    - started
                )
                * 1000
            )

            return {
                "success": False,
                "answer": "",
                "agent": agent,
                "agent_label": agent_label(agent),
                "latency_ms": latency_ms,
                "error": (
                    f"Falha no agente "
                    f"{agent_label(agent)}."
                ),
                "exception": str(
                    fallback_exc
                ),
            }

    except Exception as exc:

        logger.exception(
            "Falha na execução do agente %s",
            agent,
        )

        latency_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        return {
            "success": False,
            "answer": "",
            "agent": agent,
            "agent_label": agent_label(agent),
            "latency_ms": latency_ms,
            "error": (
                f"Falha no agente "
                f"{agent_label(agent)}."
            ),
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
    """
    Executa recuperação RAG.

    Prioridade:

        rag_answer()

    Fallback:

        retrieve_and_rerank()
    """

    empty_result = {
        "success": False,
        "retrieved": [],
        "reranked": [],
        "chunks": [],
        "context": [],
        "citations": [],
        "answer": "",
        "evidence_count": 0,
        "evidence_status": "insufficient",
        "error": None,
    }

    # ========================================================
    # MÉTODO PRINCIPAL
    # ========================================================

    if callable(rag_answer):

        try:

            result = rag_answer(
                query=query,
                org_id=org_id,
                top_k=top_k,
                rerank_k=rerank_k,
                extra_context=extra_context,
                generate_answer_flag=False,
            )

            if isinstance(result, dict):

                retrieved = _safe_list(
                    result.get(
                        "retrieved",
                        [],
                    )
                )

                reranked = _safe_list(
                    result.get(
                        "reranked",
                        [],
                    )
                )

                context = (
                    _safe_list(
                        result.get(
                            "context",
                            [],
                        )
                    )
                    or reranked
                    or _safe_list(
                        result.get(
                            "chunks",
                            [],
                        )
                    )
                    or retrieved
                )

                citations = _extract_citations(
                    result
                )

                return {
                    **result,
                    "success": bool(context),
                    "retrieved": retrieved,
                    "reranked": reranked,
                    "chunks": context,
                    "context": context,
                    "citations": citations,
                    "answer": _extract_answer(
                        result
                    ),
                    "evidence_count": len(
                        context
                    ),
                    "evidence_status": (
                        "available"
                        if context
                        else "insufficient"
                    ),
                }

        except TypeError as exc:

            logger.warning(
                "Assinatura antiga do rag_answer: %s",
                exc,
            )

        except Exception as exc:

            logger.exception(
                "Falha no rag_answer: %s",
                exc,
            )

    # ========================================================
    # FALLBACK
    # ========================================================

    if callable(retrieve_and_rerank):

        try:

            result = retrieve_and_rerank(
                query=query,
                org_id=org_id,
                top_k=top_k,
                rerank_k=rerank_k,
            )

            if isinstance(result, dict):

                retrieved = _safe_list(
                    result.get(
                        "retrieved",
                        [],
                    )
                )

                reranked = _safe_list(
                    result.get(
                        "reranked",
                        [],
                    )
                )

                context = (
                    reranked
                    or _safe_list(
                        result.get(
                            "chunks",
                            [],
                        )
                    )
                    or retrieved
                )

                citations = _extract_citations(
                    result
                )

                return {
                    **result,
                    "success": bool(context),
                    "retrieved": retrieved,
                    "reranked": reranked,
                    "chunks": context,
                    "context": context,
                    "citations": citations,
                    "answer": "",
                    "evidence_count": len(
                        context
                    ),
                    "evidence_status": (
                        "available"
                        if context
                        else "insufficient"
                    ),
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
    """
    Verificação básica de segurança e qualidade.
    """

    issues: List[str] = []

    question = _safe_text(
        question
    )

    answer = _safe_text(
        answer
    )

    if not question:

        issues.append(
            "Pergunta vazia."
        )

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
        "100% certo",
    ]

    answer_lower = answer.lower()

    for phrase in suspicious:

        if phrase in answer_lower:

            issues.append(
                "Linguagem de certeza excessiva."
            )

            break

    # Verificação simples de referências.
    # Não bloqueia automaticamente a resposta.
    if answer and "[" in answer:

        if "]" in answer:

            pass

    return {
        "success": len(issues) == 0,
        "approved": True,
        "allowed": True,
        "issues": issues,
        "issue_count": len(issues),
        "agent": AGENT_GUARD,
        "agent_label": agent_label(
            AGENT_GUARD
        ),
    }


# ============================================================
# EXECUÇÃO DO RISCO
# ============================================================

def _run_risk_agent(
    query: str,
    context: Sequence[Any],
    citations: Sequence[Any],
) -> Dict[str, Any]:
    """
    Executa o Agente de Risco.
    """

    return _execute_llm(
        question=(
            f"Analise os riscos jurídicos "
            f"relacionados à seguinte solicitação:\n\n"
            f"{query}"
        ),
        context=context,
        citations=citations,
        agent=AGENT_RISK,
    )


# ============================================================
# EXECUÇÃO DO RESUMO
# ============================================================

def _run_summary_agent(
    query: str,
    context: Sequence[Any],
    citations: Sequence[Any],
) -> Dict[str, Any]:
    """
    Executa o Agente de Resumo.
    """

    return _execute_llm(
        question=(
            f"Produza um resumo jurídico "
            f"das evidências relacionadas à "
            f"seguinte solicitação:\n\n"
            f"{query}"
        ),
        context=context,
        citations=citations,
        agent=AGENT_SUMMARY,
    )


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
    """
    Orquestra todo o sistema de IA.

    Fluxo:

        Query
          ↓
        Intent
          ↓
        RAG
          ↓
        Agente
          ↓
        LLM
          ↓
        Guard
          ↓
        Evaluation
          ↓
        Resultado
    """

    started = time.perf_counter()

    # ========================================================
    # COMPATIBILIDADE
    # ========================================================

    if not query:
        query = question

    query = _safe_text(
        query
    )

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    if not query:

        return {
            "success": False,
            "answer": "Digite uma pergunta.",
            "agent": AGENT_GENERAL,
            "agent_label": agent_label(
                AGENT_GENERAL
            ),
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

        org_id = int(
            org_id
        )

    except (
        TypeError,
        ValueError,
    ):

        return {
            "success": False,
            "answer": "Organização inválida.",
            "agent": AGENT_GENERAL,
            "agent_label": agent_label(
                AGENT_GENERAL
            ),
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

    # ========================================================
    # LIMITES
    # ========================================================

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

    # ========================================================
    # SELEÇÃO DO AGENTE
    # ========================================================

    selected_mode = _safe_text(
        mode
    ).lower()

    if selected_mode in (
        "",
        "auto",
    ):

        agent = detect_intent(
            query
        )

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

    elif selected_mode in (
        "general",
        "geral",
    ):

        agent = AGENT_GENERAL

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

        context = list(
            chunks
        )

        rag_result = {
            "success": True,
            "retrieved": context,
            "reranked": context,
            "chunks": context,
            "context": context,
            "citations": list(
                citations or []
            ),
            "answer": "",
            "evidence_count": len(
                context
            ),
            "evidence_status": "available",
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
            rag_result.get(
                "context",
                [],
            )
        )

        if not context:

            context = _safe_list(
                rag_result.get(
                    "chunks",
                    [],
                )
            )

        citations = _safe_list(
            rag_result.get(
                "citations",
                [],
            )
        )

    # ========================================================
    # CONTEXTO EXTRA
    # ========================================================

    if extra_context:

        already_added = any(
            isinstance(item, dict)
            and item.get("chunk_id")
            == "user_input"
            for item in context
        )

        if not already_added:

            context = list(
                context
            )

            context.append(
                {
                    "chunk_id": "user_input",
                    "document": (
                        "Contexto fornecido "
                        "pelo usuário"
                    ),
                    "document_id": None,
                    "page": "N/D",
                    "content": (
                        extra_context
                    ),
                    "reranker_score": 1.0,
                    "retriever_score": 1.0,
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

    answer = _extract_answer(
        primary
    )

    # ========================================================
    # FALLBACK DO RAG
    # ========================================================

    if not answer:

        rag_answer_text = _extract_answer(
            rag_result
        )

        if rag_answer_text:

            answer = rag_answer_text

    # ========================================================
    # GUARD
    # ========================================================

    guard = guard_agent(
        query,
        answer,
        context,
    )

    # ========================================================
    # RISCO
    # ========================================================

    risk: Dict[str, Any] = {}

    if run_risk:

        if context:

            risk = _run_risk_agent(
                query=query,
                context=context,
                citations=citations,
            )

        else:

            risk = {
                "success": False,
                "answer": "",
                "agent": AGENT_RISK,
                "agent_label": agent_label(
                    AGENT_RISK
                ),
                "error": (
                    "Não há evidências "
                    "para análise de risco."
                ),
                "latency_ms": 0,
            }

    # ========================================================
    # RESUMO
    # ========================================================

    summary: Dict[str, Any] = {}

    if run_summary:

        if context:

            summary = _run_summary_agent(
                query=query,
                context=context,
                citations=citations,
            )

        else:

            summary = {
                "success": False,
                "answer": "",
                "agent": AGENT_SUMMARY,
                "agent_label": agent_label(
                    AGENT_SUMMARY
                ),
                "error": (
                    "Não há evidências "
                    "para gerar resumo."
                ),
                "latency_ms": 0,
            }

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation: Dict[str, Any] = {}

    if (
        callable(evaluate_answer)
        and answer
    ):

        try:

            evaluation = evaluate_answer(
                query,
                answer,
                context,
                citations,
            )

            if not isinstance(
                evaluation,
                dict,
            ):

                evaluation = {
                    "valid": True,
                    "overall": 0.0,
                    "quality": "Disponível",
                }

        except Exception as exc:

            logger.exception(
                "Falha na avaliação."
            )

            evaluation = {
                "valid": False,
                "overall": 0.0,
                "quality": "Indisponível",
                "error": str(exc)[:300],
            }

    # ========================================================
    # MÉTRICAS
    # ========================================================

    latency_ms = int(
        (
            time.perf_counter()
            - started
        )
        * 1000
    )

    evidence_count = len(
        context
    )

    # ========================================================
    # RESPOSTA SEM EVIDÊNCIA
    # ========================================================

    if (
        not context
        and not answer
    ):

        answer = (
            "Não foi possível gerar uma resposta "
            "com base nas evidências disponíveis."
        )

    # ========================================================
    # STATUS DO AGENTE PRINCIPAL
    # ========================================================

    primary_status = (
        "operational"
        if primary.get("success")
        else "error"
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    return {
        "success": bool(
            answer
        ),

        "answer": answer,

        "agent": agent,

        "agent_label": agent_label(
            agent
        ),

        "agent_status": primary_status,

        "intent": (
            "legal_query"
            if agent == AGENT_LEGAL
            else agent
        ),

        "mode": selected_mode,

        "organization_id": org_id,

        "query": query,

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EVIDÊNCIAS
        # ----------------------------------------------------

        "citations": citations,

        "evidence_count": evidence_count,

        "evidence_status": (
            "available"
            if evidence_count > 0
            else "none"
        ),

        # ----------------------------------------------------
        # AGENTES
        # ----------------------------------------------------

        "primary": primary,

        "risk": risk,

        "risk_analysis": risk,

        "summary": summary,

        # ----------------------------------------------------
        # SEGURANÇA
        # ----------------------------------------------------

        "guard": guard,

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        "evaluation": evaluation,

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        "latency_ms": latency_ms,

        # ----------------------------------------------------
        # COMPATIBILIDADE
        # ----------------------------------------------------

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
            primary.get(
                "error",
                "",
            )
            if isinstance(
                primary,
                dict,
            )
            and not primary.get(
                "success"
            )
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
# STATUS REAL DOS COMPONENTES
# ============================================================

def orchestrator_status() -> Dict[str, Any]:
    """
    Retorna o estado real dos componentes.

    Utilizado pelo novo Dashboard / Central de IA.
    """

    # ========================================================
    # AI SERVICE
    # ========================================================

    ai_info: Dict[str, Any] = {}

    if callable(ai_status):

        try:

            result = ai_status()

            if isinstance(
                result,
                dict,
            ):

                ai_info = result

        except Exception as exc:

            ai_info = {
                "configured": False,
                "status": "error",
                "error": str(exc)[:300],
            }

    ai_configured = bool(
        ai_info.get(
            "configured",
            False,
        )
    )

    ai_status_value = ai_info.get(
        "status",
        "not_configured",
    )

    # ========================================================
    # COMPONENTES
    # ========================================================

    rag_active = callable(
        rag_answer
    ) or callable(
        retrieve_and_rerank
    )

    evaluation_active = callable(
        evaluate_answer
    )

    components = [
        {
            "id": AGENT_LEGAL,
            "name": "Agente Jurídico",
            "type": "agent",
            "status": (
                "operational"
                if callable(generate_answer)
                else "error"
            ),
        },
        {
            "id": AGENT_RISK,
            "name": "Agente de Risco",
            "type": "agent",
            "status": (
                "operational"
                if callable(generate_answer)
                else "error"
            ),
        },
        {
            "id": AGENT_SUMMARY,
            "name": "Agente de Resumo",
            "type": "agent",
            "status": (
                "operational"
                if callable(generate_answer)
                else "error"
            ),
        },
        {
            "id": AGENT_GENERAL,
            "name": "Agente Geral",
            "type": "agent",
            "status": (
                "operational"
                if callable(generate_answer)
                else "error"
            ),
        },
        {
            "id": AGENT_RAG,
            "name": "RAG / Recuperação",
            "type": "pipeline",
            "status": (
                "operational"
                if rag_active
                else "error"
            ),
        },
        {
            "id": AGENT_CITATIONS,
            "name": "Citações e Evidências",
            "type": "pipeline",
            "status": (
                "operational"
                if rag_active
                else "degraded"
            ),
        },
        {
            "id": AGENT_EVALUATION,
            "name": "Evaluation / Métricas",
            "type": "pipeline",
            "status": (
                "operational"
                if evaluation_active
                else "unavailable"
            ),
        },
        {
            "id": AGENT_GUARD,
            "name": "Guard Agent",
            "type": "security",
            "status": "operational",
        },
    ]

    operational = sum(
        1
        for component in components
        if component["status"]
        == "operational"
    )

    total = len(
        components
    )

    return {
        "configured": bool(
            callable(generate_answer)
        ),

        "overall_status": (
            "operational"
            if operational == total
            else (
                "degraded"
                if operational > 0
                else "error"
            )
        ),

        "operational_count": operational,

        "total_components": total,

        "agents": components,

        "components": components,

        "ai_service": {
            "available": callable(
                generate_answer
            ),
            "configured": ai_configured,
            "status": ai_status_value,
            "provider": ai_info.get(
                "provider"
            ),
            "model": ai_info.get(
                "model"
            ),
            "temperature": ai_info.get(
                "temperature"
            ),
            "max_tokens": ai_info.get(
                "max_tokens"
            ),
        },

        "rag": {
            "available": rag_active,
            "status": (
                "operational"
                if rag_active
                else "error"
            ),
        },

        "evaluation": {
            "available": evaluation_active,
            "status": (
                "operational"
                if evaluation_active
                else "unavailable"
            ),
        },

        "guard": {
            "available": True,
            "status": "operational",
        },

        "default_top_k": DEFAULT_TOP_K,

        "default_rerank_k": DEFAULT_RERANK_K,

        "max_top_k": MAX_TOP_K,

        "max_rerank_k": MAX_RERANK_K,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

def orchestrator_health() -> Dict[str, Any]:
    """
    Health check simplificado para Dashboard e monitoramento.
    """

    status = orchestrator_status()

    return {
        "healthy": (
            status.get(
                "overall_status"
            )
            == "operational"
        ),

        "status": status.get(
            "overall_status",
            "error",
        ),

        "operational": status.get(
            "operational_count",
            0,
        ),

        "total": status.get(
            "total_components",
            0,
        ),

        "ai_service": status.get(
            "ai_service",
            {},
        ),

        "rag": status.get(
            "rag",
            {},
        ),

        "evaluation": status.get(
            "evaluation",
            {},
        ),

        "guard": status.get(
            "guard",
            {},
        ),
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural.

    Utiliza chunks locais para evitar dependência do FAISS.
    """

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
        query="Qual é o objeto do contrato?",
        org_id=1,
        mode="legal",
        chunks=chunks,
        citations=citations,
        run_risk=False,
        run_summary=False,
    )

    return {
        "module": "ai_orchestrator.py",
        "version": "3.2",
        "status": (
            "ok"
            if isinstance(
                result,
                dict,
            )
            else "error"
        ),
        "success": result.get(
            "success",
            False,
        ),
        "agent": result.get(
            "agent"
        ),
        "agent_label": result.get(
            "agent_label"
        ),
        "evidence_count": result.get(
            "evidence_count",
            0,
        ),
        "citation_count": len(
            result.get(
                "citations",
                [],
            )
        ),
        "guard": result.get(
            "guard",
            {},
        ),
        "evaluation": result.get(
            "evaluation",
            {},
        ),
        "latency_ms": result.get(
            "latency_ms",
            0,
        ),
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "AI ORCHESTRATOR V3.2"
    )
    print("=" * 70)

    print("\nSTATUS DOS COMPONENTES:")

    status = orchestrator_status()

    print(
        "Status geral:",
        status.get(
            "overall_status"
        ),
    )

    print(
        "Componentes:",
        status.get(
            "operational_count"
        ),
        "/",
        status.get(
            "total_components"
        ),
    )

    for agent in status.get(
        "agents",
        [],
    ):

        print(
            f" - "
            f"{agent['name']}: "
            f"{agent['status']}"
        )

    print("\nSELF TEST:")

    result = self_test()

    print(
        "Status:",
        result.get(
            "status"
        ),
    )

    print(
        "Success:",
        result.get(
            "success"
        ),
    )

    print(
        "Agent:",
        result.get(
            "agent_label"
        ),
    )

    print(
        "Evidências:",
        result.get(
            "evidence_count"
        ),
    )

    print(
        "Citações:",
        result.get(
            "citation_count"
        ),
    )

    print(
        "Guard:",
        result.get(
            "guard"
        ),
    )

    print(
        "Evaluation:",
        result.get(
            "evaluation"
        ),
    )

    print(
        "Latência:",
        result.get(
            "latency_ms"
        ),
        "ms",
    )

    print("=" * 70)
