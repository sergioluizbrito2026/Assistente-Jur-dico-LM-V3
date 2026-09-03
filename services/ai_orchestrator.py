from services.agents import legal_agent
from services.agents import risk_agent
from services.agents import summary_agent
from services.agents import guard_agent

"""
AI Orchestrator - Assistente Jurídico IA SaaS V3

Responsável por:
- identificar o tipo de solicitação;
- selecionar o agente adequado;
- executar o pipeline RAG;
- chamar o agente especializado;
- retornar resposta padronizada;
- manter compatibilidade com o app atual.

Arquitetura:

Streamlit
    ↓
AI Orchestrator
    ↓
Agent
    ↓
RAG Pipeline
    ↓
Retriever
    ↓
Reranker
    ↓
Evidence Gate
    ↓
AI Service
    ↓
LLM
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from services.rag_pipeline import rag_answer


# ============================================================
# LOG
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEFAULT_TOP_K = 8
DEFAULT_RERANK_K = 5

MAX_TOP_K = 20
MAX_RERANK_K = 10


# ============================================================
# TIPOS DE AGENTE
# ============================================================

AGENT_LEGAL = "legal"
AGENT_RISK = "risk"
AGENT_SUMMARY = "summary"
AGENT_GENERAL = "general"


# ============================================================
# UTILITÁRIOS
# ============================================================

def _safe_int(
    value: Any,
    default: int,
    minimum: int = 1,
    maximum: int = 100,
) -> int:
    """
    Converte valores para inteiro com limites de segurança.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(value, maximum))


def _normalize(text: str) -> str:
    """
    Normaliza texto para classificação simples.
    """
    return " ".join((text or "").lower().strip().split())


# ============================================================
# DETECÇÃO DE INTENÇÃO
# ============================================================

def detect_intent(query: str) -> str:
    """
    Identifica de forma simples o tipo de solicitação.

    Em uma próxima etapa podemos substituir esta classificação
    por um classificador baseado em LLM.
    """

    text = _normalize(query)

    if not text:
        return AGENT_GENERAL

    risk_keywords = [
        "risco",
        "riscos",
        "perigo",
        "ameaça",
        "vulnerabilidade",
        "exposição",
        "penalidade",
        "multa",
        "cláusula problemática",
        "clausula problemática",
        "ponto crítico",
        "pontos críticos",
        "fragilidade",
        "passivo",
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
    ]

    if any(keyword in text for keyword in risk_keywords):
        return AGENT_RISK

    if any(keyword in text for keyword in summary_keywords):
        return AGENT_SUMMARY

    if any(keyword in text for keyword in legal_keywords):
        return AGENT_LEGAL

    return AGENT_GENERAL


# ============================================================
# INSTRUÇÕES DOS AGENTES
# ============================================================

def _agent_instruction(agent: str) -> str:
    """
    Retorna instruções específicas para cada agente.
    """

    instructions = {

        AGENT_LEGAL: """
Você atua como Agente Jurídico.

Analise a solicitação com base nas evidências recuperadas.

Regras:
- não invente informações;
- diferencie fatos encontrados de interpretação;
- cite as fontes utilizando [1], [2], [3] etc.;
- quando a evidência for insuficiente, informe claramente;
- destaque cláusulas, obrigações, direitos e pontos relevantes;
- não apresente a resposta como decisão jurídica definitiva.
""",

        AGENT_RISK: """
Você atua como Agente de Análise de Risco Jurídico.

Identifique, somente com base nas evidências disponíveis:

1. riscos identificados;
2. evidências que sustentam cada risco;
3. gravidade estimada;
4. possíveis impactos;
5. lacunas de informação;
6. recomendações para análise humana.

Não invente riscos que não estejam sustentados pelos documentos.

Classifique a gravidade como:
- Crítico
- Alto
- Médio
- Baixo

Sempre utilize citações [1], [2], [3] etc.
""",

        AGENT_SUMMARY: """
Você atua como Agente de Resumo Jurídico.

Produza um resumo objetivo e profissional.

Estruture preferencialmente em:

### Resumo executivo
### Principais pontos
### Obrigações
### Riscos ou pontos de atenção
### Evidências

Não acrescente fatos que não estejam presentes nos documentos.

Utilize citações [1], [2], [3] etc.
""",

        AGENT_GENERAL: """
Você atua como Assistente Jurídico Geral.

Responda de forma objetiva e profissional.

Quando a pergunta depender de documentos:
- utilize somente as evidências recuperadas;
- não invente informações;
- cite as fontes utilizando [1], [2], [3] etc.;
- informe quando não houver evidência suficiente.

Não trate a resposta como decisão jurídica definitiva.
""",
    }

    return instructions.get(agent, instructions[AGENT_GENERAL])


# ============================================================
# NOME AMIGÁVEL DO AGENTE
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
# EXECUÇÃO DO ORQUESTRADOR
# ============================================================

def orchestrate(
    query: str,
    org_id: int,
    mode: str = "auto",
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    Executa o fluxo completo da IA.

    Parameters
    ----------
    query:
        Pergunta do usuário.

    org_id:
        Organização/tenant.

    mode:
        auto, legal, risk, summary ou general.

    top_k:
        Quantidade de documentos recuperados.

    rerank_k:
        Quantidade de documentos após reranking.

    extra_context:
        Texto adicional fornecido pelo usuário.

    Returns
    -------
    Dict com resposta, agente, evidências, citações e métricas.
    """

    started_at = time.perf_counter()

    query = (query or "").strip()

    if not query:
        return {
            "success": False,
            "answer": "Digite uma pergunta para iniciar a análise.",
            "agent": AGENT_GENERAL,
            "agent_label": agent_label(AGENT_GENERAL),
            "intent": AGENT_GENERAL,
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "empty",
            "latency_ms": 0,
            "error": "empty_query",
        }

    # --------------------------------------------------------
    # VALIDAR ORGANIZAÇÃO
    # --------------------------------------------------------

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
            "evidence_count": 0,
            "evidence_status": "error",
            "latency_ms": 0,
            "error": "invalid_org_id",
        }

    # --------------------------------------------------------
    # LIMITES
    # --------------------------------------------------------

    top_k = _safe_int(
        top_k,
        DEFAULT_TOP_K,
        minimum=1,
        maximum=MAX_TOP_K,
    )

    rerank_k = _safe_int(
        rerank_k,
        DEFAULT_RERANK_K,
        minimum=1,
        maximum=MAX_RERANK_K,
    )

    if rerank_k > top_k:
        rerank_k = top_k

    # --------------------------------------------------------
    # IDENTIFICAR AGENTE
    # --------------------------------------------------------

    mode = (mode or "auto").lower().strip()

    valid_modes = {
        AGENT_LEGAL,
        AGENT_RISK,
        AGENT_SUMMARY,
        AGENT_GENERAL,
    }

    if mode == "auto":
        agent = detect_intent(query)
    elif mode in valid_modes:
        agent = mode
    else:
        agent = AGENT_GENERAL

    instruction = _agent_instruction(agent)

    logger.info(
        "AI Orchestrator | org=%s | agent=%s | query_length=%s",
        org_id,
        agent,
        len(query),
    )

    # --------------------------------------------------------
    # EXECUTAR RAG
    # --------------------------------------------------------
try:
    # EXECUÇÃO DO AGENTE ESPECIALIZADO

    if agent == AGENT_LEGAL:
        result = legal_agent.run(
            query=query,
            org_id=org_id,
            top_k=top_k,
            rerank_k=rerank_k,
            extra_context=extra_context,
        )

    elif agent == AGENT_RISK:
        result = risk_agent.run(
            query=query,
            org_id=org_id,
            top_k=top_k,
            rerank_k=rerank_k,
            extra_context=extra_context,
        )

    elif agent == AGENT_SUMMARY:
        result = summary_agent.run(
            query=query,
            org_id=org_id,
            top_k=top_k,
            rerank_k=rerank_k,
            extra_context=extra_context,
        )

    else:
        # Perguntas gerais continuam usando o RAG padrão
        result = rag_answer(
            query=query,
            org_id=org_id,
            top_k=top_k,
            rerank_k=rerank_k,
            extra_context=extra_context,
            agent_instruction=instruction,
        )

except Exception as exc:
    logger.exception("Erro no AI Orchestrator.")

    latency_ms = int(
        (time.perf_counter() - started_at) * 1000
    )

    return {
        "success": False,
        "answer": (
            "Ocorreu um erro durante a análise. "
            "Tente novamente."
        ),
        "agent": agent,
        "agent_label": agent_label(agent),
        "intent": agent,
        "citations": [],
        "context": [],
        "evidence_count": 0,
        "evidence_status": "error",
        "latency_ms": latency_ms,
        "error": str(exc),
    }

    # --------------------------------------------------------
    # NORMALIZAR RESULTADO
    # --------------------------------------------------------

    if not isinstance(result, dict):
        result = {
            "answer": str(result or ""),
            "citations": [],
            "context": [],
        }

    answer = result.get("answer", "") or ""

    citations = result.get("citations", []) or []

    context = result.get("context", []) or []

    evidence_count = result.get(
        "evidence_count",
        len(context),
    )

    evidence_status = result.get(
        "evidence_status",
        "available" if evidence_count else "none",
    )

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    latency_ms = int(
        (time.perf_counter() - started_at) * 1000
    )

    return {
        **result,

        "success": True,

        "agent": agent,

        "agent_label": agent_label(agent),

        "intent": agent,

        "latency_ms": latency_ms,

        "evidence_count": evidence_count,

        "evidence_status": evidence_status,

        "citations": citations,

        "context": context,

        "query": query,

        "organization_id": org_id,
    }


# ============================================================
# ATALHOS PARA O APP
# ============================================================

def legal_analysis(
    query: str,
    org_id: int,
    **kwargs,
) -> Dict[str, Any]:
    """
    Força o uso do Agente Jurídico.
    """

    return orchestrate(
        query=query,
        org_id=org_id,
        mode=AGENT_LEGAL,
        **kwargs,
    )


def risk_analysis(
    query: str,
    org_id: int,
    **kwargs,
) -> Dict[str, Any]:
    """
    Força o uso do Agente de Risco.
    """

    return orchestrate(
        query=query,
        org_id=org_id,
        mode=AGENT_RISK,
        **kwargs,
    )


def summarize(
    query: str,
    org_id: int,
    **kwargs,
) -> Dict[str, Any]:
    """
    Força o uso do Agente de Resumo.
    """

    return orchestrate(
        query=query,
        org_id=org_id,
        mode=AGENT_SUMMARY,
        **kwargs,
    )


# ============================================================
# STATUS
# ============================================================

def orchestrator_status() -> Dict[str, Any]:
    """
    Retorna informações básicas do orquestrador.
    """

    return {
        "configured": True,
        "agents": [
            AGENT_LEGAL,
            AGENT_RISK,
            AGENT_SUMMARY,
            AGENT_GENERAL,
        ],
        "default_top_k": DEFAULT_TOP_K,
        "default_rerank_k": DEFAULT_RERANK_K,
        "max_top_k": MAX_TOP_K,
        "max_rerank_k": MAX_RERANK_K,
    }
