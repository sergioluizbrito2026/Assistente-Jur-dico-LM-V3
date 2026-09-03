"""
Risk Agent - Assistente Jurídico IA SaaS V3

Agente especializado em identificação e análise de riscos
jurídicos com base em evidências recuperadas pelo RAG.
"""

from typing import Any, Dict

from services.rag_pipeline import build_agent_context
from services.ai import generate_answer


# ============================================================
# CONFIGURAÇÕES
# ============================================================

AGENT_NAME = "risk_agent"

AGENT_LABEL = "Agente de Risco"

DEFAULT_TOP_K = 10
DEFAULT_RERANK_K = 5


# ============================================================
# INSTRUÇÕES DO AGENTE
# ============================================================

RISK_AGENT_INSTRUCTION = """
Você é um Agente Especializado em Análise de Risco Jurídico.

Sua função é identificar riscos e pontos de atenção exclusivamente
com base nas evidências fornecidas pelos documentos.

REGRAS OBRIGATÓRIAS:

1. Nunca invente fatos.

2. Nunca invente artigos de lei.

3. Nunca invente jurisprudência.

4. Nunca invente decisões judiciais.

5. Nunca invente números de processos.

6. Todo risco identificado deve possuir uma evidência
   documental correspondente.

7. Diferencie claramente:
   - fato;
   - risco;
   - interpretação;
   - impacto;
   - recomendação.

8. Se não existir evidência suficiente para identificar um risco,
   informe isso claramente.

9. Não transforme uma possibilidade em fato.

10. Não apresente a análise como decisão jurídica definitiva.

11. Utilize as referências [1], [2], [3] etc. para indicar
    as fontes utilizadas.

CLASSIFICAÇÃO DE GRAVIDADE:

CRÍTICO:
Risco potencialmente grave e que exige atenção prioritária.

ALTO:
Risco relevante que pode gerar impacto significativo.

MÉDIO:
Risco que merece acompanhamento e análise.

BAIXO:
Ponto de atenção com impacto potencial limitado.

ESTRUTURA DA RESPOSTA:

### Resumo de Riscos

Apresente uma visão geral dos principais riscos encontrados.

### Riscos Identificados

Para cada risco apresente:

- Risco
- Gravidade
- Evidência
- Impacto potencial
- Fonte

### Pontos de Atenção

Liste cláusulas, obrigações, prazos ou condições
que merecem revisão.

### Lacunas

Informe quais informações importantes não foram encontradas
nos documentos.

### Recomendações

Apresente recomendações baseadas exclusivamente
nas evidências disponíveis.

Sempre utilize citações [1], [2], [3] etc.
"""


# ============================================================
# UTILITÁRIO
# ============================================================

def _safe_text(value: Any) -> str:
    """
    Garante que o valor seja tratado como texto.
    """
    return str(value or "").strip()


# ============================================================
# EXECUÇÃO DO AGENTE
# ============================================================

def run(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    Executa o Agente de Risco.

    Fluxo:

        Pergunta
           ↓
        RAG
           ↓
        Evidências
           ↓
        Análise de risco
           ↓
        LLM
           ↓
        Resultado estruturado
    """

    query = _safe_text(query)

    if not query:
        return {
            "success": False,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Informe uma pergunta para iniciar "
                "a análise de riscos."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "insufficient",
            "answer_generated": False,
            "error": {
                "type": "empty_query",
                "message": "A pergunta não foi informada.",
            },
        }

    # --------------------------------------------------------
    # 1. Recuperar evidências
    # --------------------------------------------------------

    try:

        rag_result = build_agent_context(
            query=query,
            org_id=org_id,
            top_k=top_k,
            rerank_k=rerank_k,
            extra_context=extra_context,
        )

    except Exception as exc:

        return {
            "success": False,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não foi possível recuperar as evidências "
                "necessárias para a análise de risco."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "error",
            "answer_generated": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            },
        }

    context = rag_result.get(
        "context",
        [],
    )

    citations = rag_result.get(
        "citations",
        [],
    )

    evidence_count = rag_result.get(
        "evidence_count",
        len(context),
    )

    evidence_status = rag_result.get(
        "evidence_status",
        "available" if context else "insufficient",
    )

    # --------------------------------------------------------
    # 2. Evidence Gate
    # --------------------------------------------------------

    if not context:

        return {
            "success": True,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não há evidências suficientes nos documentos "
                "disponibilizados para realizar uma análise "
                "de risco confiável."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "insufficient",
            "answer_generated": False,
            "error": None,
        }

    # --------------------------------------------------------
    # 3. Gerar análise
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            query=query,
            chunks=context,
            agent_instruction=RISK_AGENT_INSTRUCTION,
        )

    except Exception as exc:

        return {
            "success": False,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não foi possível gerar a análise "
                "de risco neste momento."
            ),
            "citations": citations,
            "context": context,
            "evidence_count": evidence_count,
            "evidence_status": evidence_status,
            "answer_generated": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            },
        }

    # --------------------------------------------------------
    # 4. Resultado
    # --------------------------------------------------------

    return {
        "success": True,

        "agent": AGENT_NAME,

        "agent_label": AGENT_LABEL,

        "answer": _safe_text(answer),

        "citations": citations,

        "context": context,

        "evidence_count": evidence_count,

        "evidence_status": evidence_status,

        "answer_generated": bool(
            _safe_text(answer)
        ),

        "query": query,

        "organization_id": org_id,

        "error": None,
    }


# ============================================================
# ALIAS
# ============================================================

def analyze(
    query: str,
    org_id: int,
    **kwargs,
) -> Dict[str, Any]:
    """
    Alias para chamadas futuras.
    """

    return run(
        query=query,
        org_id=org_id,
        **kwargs,
    )


# ============================================================
# STATUS
# ============================================================

def status() -> Dict[str, Any]:
    """
    Retorna o status do agente.
    """

    return {
        "configured": True,
        "name": AGENT_NAME,
        "label": AGENT_LABEL,
        "type": "risk",
    }
