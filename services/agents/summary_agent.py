"""
Summary Agent - Assistente Jurídico IA SaaS V3

Agente especializado em resumo executivo de documentos jurídicos.
"""

from typing import Any, Dict

from services.rag_pipeline import build_agent_context
from services.ai import generate_answer


AGENT_NAME = "summary_agent"
AGENT_LABEL = "Agente de Resumo"

DEFAULT_TOP_K = 10
DEFAULT_RERANK_K = 5


SUMMARY_AGENT_INSTRUCTION = """
Você é um Agente Especializado em Resumo Jurídico.

Sua função é produzir resumos objetivos e fiéis aos documentos
fornecidos pelo sistema.

REGRAS OBRIGATÓRIAS:

1. Utilize somente as evidências fornecidas.

2. Não invente fatos.

3. Não invente artigos de lei.

4. Não invente jurisprudência.

5. Não invente números de processos.

6. Não acrescente informações externas que não estejam nas evidências.

7. Diferencie fatos documentais de interpretações.

8. Quando uma informação não estiver disponível, informe claramente.

9. Preserve informações importantes como:
   - partes;
   - objeto;
   - obrigações;
   - prazos;
   - valores;
   - condições;
   - cláusulas;
   - penalidades;
   - rescisão;
   - riscos identificados.

10. Utilize [1], [2], [3] etc. para indicar as fontes.

ESTRUTURA DA RESPOSTA:

### Resumo Executivo

Apresente uma visão geral objetiva do documento.

### Principais Pontos

Liste os principais assuntos identificados.

### Obrigações e Responsabilidades

Destaque as principais obrigações das partes.

### Prazos e Condições

Informe prazos, vigência e condições relevantes,
quando presentes nas evidências.

### Pontos de Atenção

Destaque informações que merecem análise.

### Conclusão

Faça uma síntese final baseada exclusivamente
nas evidências disponíveis.

Não apresente o resumo como decisão jurídica definitiva.
"""


def _safe_text(value: Any) -> str:
    """
    Converte o valor recebido para texto seguro.
    """
    return str(value or "").strip()


def run(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    Executa o Agente de Resumo.

    Fluxo:

        Pergunta
           ↓
        RAG
           ↓
        Evidências
           ↓
        Resumo
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
            "answer": (
                "Informe o documento ou a solicitação "
                "que deseja resumir."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "insufficient",
            "answer_generated": False,
            "error": {
                "type": "empty_query",
                "message": "A solicitação não foi informada.",
            },
        }

    # ========================================================
    # 1. RECUPERAR EVIDÊNCIAS
    # ========================================================

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
                "necessárias para gerar o resumo."
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

    # ========================================================
    # 2. EVIDENCE GATE
    # ========================================================

    if not context:

        return {
            "success": True,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não há evidências suficientes nos documentos "
                "disponibilizados para gerar um resumo confiável."
            ),
            "citations": [],
            "context": [],
            "evidence_count": 0,
            "evidence_status": "insufficient",
            "answer_generated": False,
            "error": None,
        }

    # ========================================================
    # 3. GERAR RESUMO
    # ========================================================

    try:

        answer = generate_answer(
            query=query,
            chunks=context,
            agent_instruction=SUMMARY_AGENT_INSTRUCTION,
        )

    except Exception as exc:

        return {
            "success": False,
            "agent": AGENT_NAME,
            "agent_label": AGENT_LABEL,
            "answer": (
                "Não foi possível gerar o resumo "
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

    # ========================================================
    # 4. RESULTADO FINAL
    # ========================================================

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


def summarize(
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


def status() -> Dict[str, Any]:
    """
    Retorna o status do agente.
    """

    return {
        "configured": True,
        "name": AGENT_NAME,
        "label": AGENT_LABEL,
        "type": "summary",
    }
