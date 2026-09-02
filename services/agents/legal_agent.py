"""
Legal Agent - Assistente Jurídico IA SaaS V3

Agente especializado em análise jurídica baseada em evidências
recuperadas pelo pipeline RAG.
"""

from typing import Any, Dict

from services.rag_pipeline import build_agent_context
from services.ai import generate_answer


# ============================================================
# CONFIGURAÇÕES
# ============================================================

AGENT_NAME = "legal_agent"

AGENT_LABEL = "Agente Jurídico"

DEFAULT_TOP_K = 10
DEFAULT_RERANK_K = 5


# ============================================================
# INSTRUÇÕES DO AGENTE
# ============================================================

LEGAL_AGENT_INSTRUCTION = """
Você é um Agente Jurídico de apoio à análise documental.

Sua função é analisar documentos jurídicos utilizando SOMENTE
as evidências fornecidas pelo sistema.

REGRAS OBRIGATÓRIAS:

1. Não invente fatos.

2. Não invente artigos de lei.

3. Não invente jurisprudência.

4. Não invente números de processos.

5. Não invente decisões judiciais.

6. Diferencie claramente:
   - fatos encontrados;
   - interpretação;
   - riscos;
   - recomendações.

7. Sempre que utilizar uma informação recuperada dos documentos,
   indique a fonte utilizando [1], [2], [3] etc.

8. Se as evidências forem insuficientes, informe explicitamente
   que não há evidência suficiente para uma conclusão segura.

9. Não trate a resposta como decisão jurídica definitiva.

10. Recomende análise profissional quando a situação exigir
    interpretação jurídica especializada.

ESTRUTURA PREFERENCIAL:

### Análise

Explique objetivamente o que foi identificado.

### Evidências

Apresente os principais elementos encontrados nos documentos.

### Pontos de atenção

Destaque cláusulas, obrigações, prazos, condições ou inconsistências
relevantes.

### Interpretação

Separe interpretação de fatos documentais.

### Recomendação

Apresente recomendações baseadas exclusivamente nas evidências.

Sempre utilize as citações [1], [2], [3] etc.
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
    Executa o Agente Jurídico.

    Fluxo:

        Pergunta
           ↓
        RAG
           ↓
        Evidências
           ↓
        Agente Jurídico
           ↓
        LLM
           ↓
        Resposta + Citações
    """

    query = _safe_text(query)

    if not query:
        return {
            "success": False,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": "Informe uma pergunta para iniciar a análise.",
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "insufficient",
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
                "necessárias para a análise."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "error",
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
    # 2. Verificação de evidências
    # --------------------------------------------------------

    if not context:

        return {
            "success": True,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não há evidência suficiente nos documentos "
                "disponibilizados para realizar uma análise "
                "jurídica segura."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "insufficient",
            "answer_generated": False,
        }

    # --------------------------------------------------------
    # 3. Gerar análise jurídica
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            query=query,
            chunks=context,
            agent_instruction=LEGAL_AGENT_INSTRUCTION,
        )

    except Exception as exc:

        return {
            "success": False,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não foi possível gerar a análise jurídica "
                "neste momento."
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
    Alias para facilitar chamadas futuras.
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
        "type": "legal",
    }
