"""
Assistente Jurídico SaaS IA V3.2
services/ai.py

AI Service central da plataforma.

Responsabilidades:

- Gerenciar provedores LLM
- Gemini
- OpenAI
- Modo demonstração
- Prompt jurídico central
- Agentes especializados
- Compatibilidade com RAG
- Compatibilidade com AI Orchestrator
- Metadados de execução
- Latência
- Status do provedor
- Tratamento de erros
- Controle de temperatura
- Controle de tokens
- Cache dos clientes LLM

Agentes suportados:

1. Agente Jurídico
2. Agente de Risco
3. Agente de Resumo
4. Agente Geral

Compatibilidade:

    generate_answer(query, chunks)

    generate_answer(
        query,
        chunks,
        agent_instruction="..."
    )

Também aceita:

    generate_answer(prompt)

quando chamado diretamente pelo orchestrator.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import os
import time
from typing import Any, Dict, List, Sequence


# ============================================================
# LOG
# ============================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


# ============================================================
# AGENTES
# ============================================================

AGENT_LEGAL = "legal"
AGENT_RISK = "risk"
AGENT_SUMMARY = "summary"
AGENT_GENERAL = "general"


AGENT_LABELS = {
    AGENT_LEGAL: "Agente Jurídico",
    AGENT_RISK: "Agente de Risco",
    AGENT_SUMMARY: "Agente de Resumo",
    AGENT_GENERAL: "Agente Geral",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Você é o Assistente Jurídico de uma plataforma profissional
de análise documental com Inteligência Artificial.

Sua função é auxiliar profissionais do Direito na análise de
documentos, contratos, processos e evidências documentais.

REGRAS FUNDAMENTAIS:

1. Quando a pergunta depender de documentos, utilize
   prioritariamente as evidências fornecidas no contexto.

2. Nunca invente fatos.

3. Nunca invente:
   - artigos de lei;
   - jurisprudência;
   - precedentes;
   - números de processo;
   - datas;
   - valores;
   - cláusulas;
   - nomes;
   - decisões;
   - informações documentais.

4. Se não houver evidência suficiente, informe claramente:

"Não há evidência suficiente nos documentos disponibilizados."

5. Diferencie claramente:

   FATO
   INTERPRETAÇÃO
   RISCO
   RECOMENDAÇÃO

6. Nunca apresente interpretação como fato.

7. Utilize referências [1], [2], [3] etc. somente quando
   elas existirem no contexto fornecido.

8. Nunca crie uma referência inexistente.

9. Se houver informações conflitantes entre documentos,
   informe explicitamente o conflito.

10. Não trate a resposta como decisão jurídica definitiva.

11. Seja objetivo, profissional, estruturado e auditável.

12. Não utilize conhecimento externo para preencher lacunas
    documentais, salvo quando isso for explicitamente solicitado.

13. Priorize precisão e fidelidade às evidências.

14. Se a evidência não permitir concluir algo, diga isso
    explicitamente.

15. Nunca transforme uma possibilidade em certeza.

16. Não utilize linguagem de certeza absoluta quando a
    documentação não sustentar essa conclusão.

17. Sempre que possível, indique a origem da informação.

18. A resposta deve ser útil para análise profissional,
    mas não substitui a avaliação de um advogado.
"""


# ============================================================
# CONFIGURAÇÃO
# ============================================================

def _get_provider() -> str:
    """
    Retorna o provedor configurado.

    Valores suportados:

        demo
        gemini
        openai
    """

    return os.getenv(
        "LLM_PROVIDER",
        "demo",
    ).strip().lower()


def _get_model(provider: str | None = None) -> str:

    provider = provider or _get_provider()

    if provider == "gemini":

        return os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ).strip()

    if provider == "openai":

        return os.getenv(
            "OPENAI_MODEL",
            "gpt-5-mini",
        ).strip()

    return "demo"


def _get_temperature() -> float:

    try:

        value = float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0.2",
            )
        )

        return max(
            0.0,
            min(value, 1.0),
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.2


def _get_max_tokens() -> int:

    try:

        value = int(
            os.getenv(
                "LLM_MAX_TOKENS",
                "2048",
            )
        )

        return max(
            256,
            min(value, 16384),
        )

    except (
        TypeError,
        ValueError,
    ):

        return 2048


def _get_timeout() -> float:

    try:

        value = float(
            os.getenv(
                "LLM_TIMEOUT",
                "60",
            )
        )

        return max(
            5.0,
            min(value, 300.0),
        )

    except (
        TypeError,
        ValueError,
    ):

        return 60.0


# ============================================================
# UTILITÁRIOS
# ============================================================

def _safe_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    try:
        return str(value).strip()

    except Exception:
        return ""


def _safe_chunks(
    chunks: Any,
) -> List[Any]:

    if chunks is None:
        return []

    if isinstance(chunks, list):
        return chunks

    if isinstance(chunks, tuple):
        return list(chunks)

    if isinstance(chunks, Sequence) and not isinstance(
        chunks,
        (str, bytes),
    ):
        return list(chunks)

    return [chunks]


# ============================================================
# CONTEXTO
# ============================================================

def _context(
    chunks: Sequence[Any] | None,
) -> str:
    """
    Constrói o contexto documental.

    As referências [1], [2], [3] correspondem
    exatamente à ordem dos chunks.
    """

    chunks = _safe_chunks(chunks)

    if not chunks:

        return (
            "Nenhuma evidência documental "
            "foi recuperada."
        )

    parts: List[str] = []

    for index, chunk in enumerate(
        chunks,
        1,
    ):

        if isinstance(
            chunk,
            dict,
        ):

            document = (
                chunk.get("document")
                or chunk.get("document_name")
                or chunk.get("name")
                or "Documento desconhecido"
            )

            page = (
                chunk.get("page")
                if chunk.get("page") is not None
                else "N/D"
            )

            chunk_id = (
                chunk.get("chunk_id")
                or chunk.get("id")
                or "N/D"
            )

            content = (
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("page_content")
                or chunk.get("chunk")
                or ""
            )

        else:

            document = "Documento"

            page = "N/D"

            chunk_id = "N/D"

            content = _safe_text(
                chunk
            )

        content = _safe_text(
            content
        )

        if not content:
            continue

        parts.append(
            f"[{index}] "
            f"DOCUMENTO: {document} | "
            f"PÁGINA: {page} | "
            f"CHUNK: {chunk_id}\n"
            f"{content}"
        )

    if not parts:

        return (
            "Nenhuma evidência documental "
            "válida foi recuperada."
        )

    return "\n\n".join(parts)


# ============================================================
# INSTRUÇÕES DOS AGENTES
# ============================================================

def legal_agent_instruction() -> str:

    return """
Você está atuando como AGENTE JURÍDICO.

Objetivo:

Realizar análise jurídica documental baseada nas evidências
fornecidas.

Estruture a resposta, quando aplicável, em:

### Fato identificado
### Análise
### Base documental
### Pontos relevantes
### Conclusão

Regras:

- Não invente legislação.
- Não invente jurisprudência.
- Não invente fatos.
- Diferencie fato de interpretação.
- Utilize citações [N].
- Se não houver evidência suficiente, informe isso.
"""


def risk_agent_instruction() -> str:

    return """
Você está atuando como AGENTE DE RISCO JURÍDICO.

Objetivo:

Identificar riscos, inconsistências, lacunas e pontos críticos
presentes nas evidências.

Classifique cada risco, quando possível, como:

- CRÍTICO
- ALTO
- MÉDIO
- BAIXO

Estruture a resposta em:

### Riscos identificados
### Pontos críticos
### Inconsistências
### Lacunas
### Impactos possíveis
### Recomendações
### Evidências

Cada risco deve estar relacionado a uma evidência.

Não invente informações.
Utilize [N] para as fontes.
"""


def summary_agent_instruction() -> str:

    return """
Você está atuando como AGENTE DE RESUMO JURÍDICO.

Produza um resumo fiel e objetivo das evidências.

Estruture em:

### Resumo executivo

### Principais pontos

### Obrigações

### Prazos

### Valores

### Pontos de atenção

### Evidências

Ignore seções que não tenham informações suficientes.

Não invente informações.
Utilize [N] quando houver evidência correspondente.
"""


def general_agent_instruction() -> str:

    return """
Você está atuando como AGENTE GERAL.

Responda de maneira clara, profissional e objetiva.

Quando a pergunta depender dos documentos:

- utilize as evidências;
- cite as fontes;
- não invente informações;
- informe quando houver insuficiência de evidências;
- diferencie fato de interpretação.

Quando a pergunta não depender dos documentos,
responda normalmente, deixando claro quando estiver
utilizando conhecimento geral.
"""


def get_agent_instruction(
    agent: str | None,
) -> str:

    agent = _safe_text(
        agent
    ).lower()

    if agent == AGENT_LEGAL:
        return legal_agent_instruction()

    if agent == AGENT_RISK:
        return risk_agent_instruction()

    if agent == AGENT_SUMMARY:
        return summary_agent_instruction()

    return general_agent_instruction()


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    query: str,
    chunks: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    agent_instruction: str | None = None,
) -> str:
    """
    Constrói o prompt final enviado ao LLM.
    """

    query = _safe_text(
        query
    )

    base_system = (
        system_prompt
        or SYSTEM_PROMPT
    )

    context = _context(
        chunks
    )

    parts = [
        base_system,
        "",
    ]

    if agent_instruction:

        parts.extend(
            [
                "INSTRUÇÃO DO AGENTE:",
                agent_instruction.strip(),
                "",
            ]
        )

    parts.extend(
        [
            "CONTEXTO / EVIDÊNCIAS:",
            context,
            "",
            "PERGUNTA DO USUÁRIO:",
            query,
            "",
            "REGRAS DE RESPOSTA:",
            "- Baseie fatos nas evidências.",
            "- Não invente informações.",
            "- Utilize [N] para evidências.",
            "- Informe conflitos documentais.",
            "- Informe insuficiência de evidência.",
            "- Diferencie fato de interpretação.",
            "",
            "RESPONDA AGORA:",
        ]
    )

    return "\n".join(
        parts
    )


# ============================================================
# CLIENTE GEMINI
# ============================================================

@lru_cache(
    maxsize=1
)
def _get_gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return None

    try:

        from google import genai

        return genai.Client(
            api_key=api_key
        )

    except Exception as exc:

        logger.exception(
            "Erro ao inicializar Gemini: %s",
            exc,
        )

        return None


# ============================================================
# CLIENTE OPENAI
# ============================================================

@lru_cache(
    maxsize=1
)
def _get_openai_client():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        return None

    try:

        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            timeout=_get_timeout(),
        )

    except Exception as exc:

        logger.exception(
            "Erro ao inicializar OpenAI: %s",
            exc,
        )

        return None


# ============================================================
# GEMINI
# ============================================================

def _generate_gemini(
    prompt: str,
) -> str:

    client = _get_gemini_client()

    if client is None:

        raise RuntimeError(
            "Cliente Gemini não configurado."
        )

    model = _get_model(
        "gemini"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    text = getattr(
        response,
        "text",
        None,
    )

    if not text:

        raise RuntimeError(
            "O Gemini não retornou conteúdo."
        )

    return _safe_text(
        text
    )


# ============================================================
# OPENAI
# ============================================================

def _generate_openai(
    prompt: str,
) -> str:

    client = _get_openai_client()

    if client is None:

        raise RuntimeError(
            "Cliente OpenAI não configurado."
        )

    model = _get_model(
        "openai"
    )

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=_get_temperature(),

        max_tokens=_get_max_tokens(),
    )

    if not response.choices:

        raise RuntimeError(
            "A OpenAI não retornou escolhas."
        )

    message = response.choices[0].message

    content = getattr(
        message,
        "content",
        None,
    )

    if not content:

        raise RuntimeError(
            "A OpenAI não retornou conteúdo."
        )

    return _safe_text(
        content
    )


# ============================================================
# MODO DEMO
# ============================================================

def _demo_response(
    query: str = "",
    agent: str = AGENT_GENERAL,
) -> str:

    label = AGENT_LABELS.get(
        agent,
        "Agente Geral",
    )

    return f"""
### 🤖 Modo demonstração

**Agente:** {label}

O pipeline de Inteligência Artificial foi executado,
mas nenhum provedor LLM está configurado neste ambiente.

Configure um dos provedores:

`LLM_PROVIDER=gemini`

e:

`GEMINI_API_KEY`

ou:

`LLM_PROVIDER=openai`

e:

`OPENAI_API_KEY`

O RAG, as evidências, as citações e as métricas
continuam disponíveis para inspeção.

**Pergunta recebida:**

{query}
""".strip()


# ============================================================
# NORMALIZAÇÃO DE ARGUMENTOS
# ============================================================

def _prepare_request(
    query: Any,
    chunks: Any = None,
) -> tuple[str, List[Any]]:

    query_text = _safe_text(
        query
    )

    normalized_chunks = _safe_chunks(
        chunks
    )

    return (
        query_text,
        normalized_chunks,
    )


# ============================================================
# GERAÇÃO PRINCIPAL
# ============================================================

def generate_answer(
    query,
    chunks=None,
    system_prompt=None,
    agent_instruction=None,
):
    """
    Interface principal do AI Service.

    Compatibilidade:

        generate_answer(query, chunks)

    ou:

        generate_answer(
            query,
            chunks,
            agent_instruction="..."
        )

    Também suporta:

        generate_answer(prompt)

    quando o orchestrator envia um prompt completo.
    """

    query_text, chunks_list = _prepare_request(
        query,
        chunks,
    )

    if not query_text:

        return (
            "Não foi fornecida uma pergunta."
        )

    provider = _get_provider()

    # --------------------------------------------------------
    # AGENTE
    # --------------------------------------------------------

    agent = AGENT_GENERAL

    if agent_instruction:

        instruction_text = _safe_text(
            agent_instruction
        ).lower()

        if "agente jurídico" in instruction_text:
            agent = AGENT_LEGAL

        elif "agente de risco" in instruction_text:
            agent = AGENT_RISK

        elif "agente de resumo" in instruction_text:
            agent = AGENT_SUMMARY

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    # Se chunks foram fornecidos, constrói o prompt normalmente.
    #
    # Se não foram fornecidos, consideramos que o argumento
    # pode ser um prompt completo enviado pelo orchestrator.

    if chunks_list:

        prompt = build_prompt(
            query=query_text,
            chunks=chunks_list,
            system_prompt=system_prompt,
            agent_instruction=agent_instruction,
        )

    else:

        if agent_instruction:

            prompt = build_prompt(
                query=query_text,
                chunks=[],
                system_prompt=system_prompt,
                agent_instruction=agent_instruction,
            )

        else:

            # Compatibilidade com:
            #
            # generate_answer(prompt)
            #
            # usado pelo ai_orchestrator.

            prompt = query_text

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    if provider == "demo":

        return _demo_response(
            query=query_text,
            agent=agent,
        )

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    if provider == "gemini":

        return _generate_gemini(
            prompt
        )

    # --------------------------------------------------------
    # OPENAI
    # --------------------------------------------------------

    if provider == "openai":

        return _generate_openai(
            prompt
        )

    # --------------------------------------------------------
    # PROVIDER INVÁLIDO
    # --------------------------------------------------------

    raise RuntimeError(
        f"LLM_PROVIDER inválido: {provider}"
    )


# ============================================================
# RESULTADO ESTRUTURADO
# ============================================================

def generate_answer_result(
    query,
    chunks=None,
    system_prompt=None,
    agent_instruction=None,
) -> Dict[str, Any]:
    """
    Executa a IA retornando metadados.

    Útil para:

    - Dashboard
    - Auditoria
    - Métricas
    - Monitoramento
    - Agentes
    - Observabilidade
    """

    provider = _get_provider()

    model = _get_model(
        provider
    )

    started = time.perf_counter()

    try:

        answer = generate_answer(
            query=query,
            chunks=chunks,
            system_prompt=system_prompt,
            agent_instruction=agent_instruction,
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

            "answer": _safe_text(
                answer
            ),

            "provider": provider,

            "model": model,

            "latency_ms": latency_ms,

            "temperature": _get_temperature(),

            "max_tokens": _get_max_tokens(),

            "error": None,
        }

    except Exception as exc:

        latency_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        logger.exception(
            "Falha na geração da resposta."
        )

        return {
            "success": False,

            "answer": "",

            "provider": provider,

            "model": model,

            "latency_ms": latency_ms,

            "temperature": _get_temperature(),

            "max_tokens": _get_max_tokens(),

            "error": {
                "type": type(
                    exc
                ).__name__,

                "message": str(
                    exc
                )[:500],
            },
        }


# ============================================================
# EXECUÇÃO POR AGENTE
# ============================================================

def generate_agent_answer(
    query: str,
    chunks: Sequence[Any] | None = None,
    agent: str = AGENT_GENERAL,
) -> Dict[str, Any]:
    """
    Executa um agente especializado.

    Exemplo:

        generate_agent_answer(
            query,
            chunks,
            "risk"
        )
    """

    agent = _safe_text(
        agent
    ).lower()

    if agent not in AGENT_LABELS:

        agent = AGENT_GENERAL

    instruction = get_agent_instruction(
        agent
    )

    result = generate_answer_result(
        query=query,
        chunks=chunks,
        agent_instruction=instruction,
    )

    result.update(
        {
            "agent": agent,
            "agent_label": AGENT_LABELS[
                agent
            ],
        }
    )

    return result


# ============================================================
# FUNÇÕES DOS AGENTES
# ============================================================

def legal_answer(
    query: str,
    chunks: Sequence[Any] | None = None,
) -> Dict[str, Any]:

    return generate_agent_answer(
        query=query,
        chunks=chunks,
        agent=AGENT_LEGAL,
    )


def risk_answer(
    query: str,
    chunks: Sequence[Any] | None = None,
) -> Dict[str, Any]:

    return generate_agent_answer(
        query=query,
        chunks=chunks,
        agent=AGENT_RISK,
    )


def summary_answer(
    query: str,
    chunks: Sequence[Any] | None = None,
) -> Dict[str, Any]:

    return generate_agent_answer(
        query=query,
        chunks=chunks,
        agent=AGENT_SUMMARY,
    )


def general_answer(
    query: str,
    chunks: Sequence[Any] | None = None,
) -> Dict[str, Any]:

    return generate_agent_answer(
        query=query,
        chunks=chunks,
        agent=AGENT_GENERAL,
    )


# ============================================================
# STATUS
# ============================================================

def ai_status() -> Dict[str, Any]:
    """
    Retorna o estado atual do AI Service.
    """

    provider = _get_provider()

    model = _get_model(
        provider
    )

    if provider == "gemini":

        configured = bool(
            os.getenv(
                "GEMINI_API_KEY"
            )
        )

    elif provider == "openai":

        configured = bool(
            os.getenv(
                "OPENAI_API_KEY"
            )
        )

    else:

        configured = False

    if configured:

        status = "connected"

    elif provider == "demo":

        status = "demo"

    else:

        status = "not_configured"

    return {
        "configured": configured,

        "provider": provider,

        "model": model,

        "status": status,

        "temperature": _get_temperature(),

        "max_tokens": _get_max_tokens(),

        "timeout": _get_timeout(),

        "agents": [
            {
                "id": AGENT_LEGAL,
                "name": AGENT_LABELS[
                    AGENT_LEGAL
                ],
                "status": "active",
            },
            {
                "id": AGENT_RISK,
                "name": AGENT_LABELS[
                    AGENT_RISK
                ],
                "status": "active",
            },
            {
                "id": AGENT_SUMMARY,
                "name": AGENT_LABELS[
                    AGENT_SUMMARY
                ],
                "status": "active",
            },
            {
                "id": AGENT_GENERAL,
                "name": AGENT_LABELS[
                    AGENT_GENERAL
                ],
                "status": "active",
            },
        ],
    }


# ============================================================
# LIMPEZA DE CACHE
# ============================================================

def clear_ai_cache() -> Dict[str, Any]:
    """
    Limpa os clientes LLM em memória.

    Útil após alteração de:

    - API Key
    - Provider
    - Modelo
    - Configuração
    """

    _get_gemini_client.cache_clear()

    _get_openai_client.cache_clear()

    return {
        "success": True,
        "message": "Cache dos clientes LLM limpo.",
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural.

    Não exige API Key.
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

    try:

        prompt = build_prompt(
            query="Qual é o objeto do contrato?",
            chunks=chunks,
            agent_instruction=(
                legal_agent_instruction()
            ),
        )

        return {
            "status": "ok",

            "module": "services.ai",

            "provider": _get_provider(),

            "model": _get_model(),

            "prompt_generated": bool(
                prompt
            ),

            "prompt_length": len(
                prompt
            ),

            "agents": list(
                AGENT_LABELS.keys()
            ),
        }

    except Exception as exc:

        return {
            "status": "error",

            "module": "services.ai",

            "error": {
                "type": type(
                    exc
                ).__name__,

                "message": str(
                    exc
                )[:500],
            },
        }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "AI SERVICE V3.2 - SELF TEST"
    )

    print("=" * 70)

    result = self_test()

    print(
        "Status:",
        result.get("status"),
    )

    print(
        "Provider:",
        result.get("provider"),
    )

    print(
        "Model:",
        result.get("model"),
    )

    print(
        "Prompt:",
        result.get("prompt_generated"),
    )

    print(
        "Agentes:",
        result.get("agents"),
    )

    print("=" * 70)
