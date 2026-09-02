from functools import lru_cache
import os
import time
from typing import Any


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SYSTEM_PROMPT = """
Você é um Assistente Jurídico de apoio à análise documental.

REGRAS FUNDAMENTAIS:

1. Quando a pergunta depender de documentos, utilize somente as
   evidências fornecidas no contexto.

2. Nunca invente fatos, artigos de lei, jurisprudência, precedentes,
   números de processo, datas, valores ou informações que não estejam
   nas evidências.

3. Se não houver evidência suficiente, informe explicitamente:
   "Não há evidência suficiente nos documentos disponibilizados."

4. Diferencie claramente:
   - Fato encontrado no documento
   - Interpretação
   - Risco
   - Recomendação

5. Não apresente interpretação como fato.

6. Utilize [1], [2], [3] etc. para indicar as fontes utilizadas.

7. Não crie referências ou citações que não existam no contexto.

8. Se as fontes apresentarem informações conflitantes, informe
   explicitamente a existência do conflito.

9. Não trate a resposta como decisão jurídica definitiva.

10. Seja objetivo, estruturado, profissional e auditável.

11. Não utilize conhecimento externo para preencher lacunas documentais,
    salvo quando isso for explicitamente solicitado.

12. Priorize precisão e fidelidade às evidências em vez de especulação.
"""


# ============================================================
# CONFIGURAÇÕES DO SISTEMA
# ============================================================

def _get_provider():
    return os.getenv(
        "LLM_PROVIDER",
        "demo",
    ).strip().lower()


def _get_model(provider=None):

    provider = provider or _get_provider()

    if provider == "gemini":
        return os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

    if provider == "openai":
        return os.getenv(
            "OPENAI_MODEL",
            "gpt-5-mini",
        )

    return "demo"


def _get_temperature():
    try:
        return float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0.2",
            )
        )
    except (TypeError, ValueError):
        return 0.2


def _get_max_tokens():

    try:
        return int(
            os.getenv(
                "LLM_MAX_TOKENS",
                "2048",
            )
        )

    except (TypeError, ValueError):
        return 2048


def _get_timeout():

    try:
        return float(
            os.getenv(
                "LLM_TIMEOUT",
                "60",
            )
        )

    except (TypeError, ValueError):
        return 60.0


# ============================================================
# CONTEXTO
# ============================================================

def _context(chunks):
    """
    Constrói o contexto utilizado pelo LLM.

    As referências [1], [2], [3] correspondem à ordem
    dos chunks enviados.
    """

    if not chunks:
        return "Nenhuma evidência documental foi recuperada."

    parts = []

    for i, chunk in enumerate(
        chunks,
        1,
    ):

        document = chunk.get(
            "document",
            "Desconhecido",
        )

        page = chunk.get(
            "page",
            "N/D",
        )

        chunk_id = chunk.get(
            "chunk_id",
            "N/D",
        )

        content = (
            chunk.get(
                "content",
                "",
            )
            or ""
        ).strip()

        parts.append(
            f"[{i}] "
            f"DOCUMENTO: {document} | "
            f"PÁGINA: {page} | "
            f"CHUNK: {chunk_id}\n"
            f"{content}"
        )

    return "\n\n".join(parts)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    query,
    chunks,
    system_prompt=None,
    agent_instruction=None,
):
    """
    Constrói o prompt final.

    agent_instruction permite que futuros agentes especializados
    utilizem o mesmo AI Service.

    Exemplo:

        Legal Agent
        Risk Agent
        Summary Agent
    """

    base_system = (
        system_prompt
        or SYSTEM_PROMPT
    )

    context = _context(chunks)

    prompt_parts = [
        base_system,
    ]

    if agent_instruction:

        prompt_parts.extend(
            [
                "",
                "INSTRUÇÃO ESPECIALIZADA DO AGENTE:",
                agent_instruction,
            ]
        )

    prompt_parts.extend(
        [
            "",
            "CONTEXTO / EVIDÊNCIAS:",
            context,
            "",
            "PERGUNTA DO USUÁRIO:",
            query,
            "",
            "IMPORTANTE:",
            "Baseie as afirmações factuais nas evidências.",
            "Utilize as referências [N] quando aplicável.",
            "Não invente informações ausentes.",
        ]
    )

    return "\n".join(
        prompt_parts
    )


# ============================================================
# CLIENTE GEMINI
# ============================================================

@lru_cache(maxsize=1)
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

    except Exception:
        return None


# ============================================================
# CLIENTE OPENAI
# ============================================================

@lru_cache(maxsize=1)
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

    except Exception:
        return None


# ============================================================
# GEMINI
# ============================================================

def _generate_gemini(
    query,
    chunks,
    system_prompt=None,
    agent_instruction=None,
):

    client = _get_gemini_client()

    if client is None:

        raise RuntimeError(
            "Cliente Gemini não configurado."
        )

    model = _get_model(
        "gemini"
    )

    prompt = build_prompt(
        query=query,
        chunks=chunks,
        system_prompt=system_prompt,
        agent_instruction=agent_instruction,
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

    return text.strip()


# ============================================================
# OPENAI
# ============================================================

def _generate_openai(
    query,
    chunks,
    system_prompt=None,
    agent_instruction=None,
):

    client = _get_openai_client()

    if client is None:

        raise RuntimeError(
            "Cliente OpenAI não configurado."
        )

    model = _get_model(
        "openai"
    )

    base_system = (
        system_prompt
        or SYSTEM_PROMPT
    )

    prompt = build_prompt(
        query=query,
        chunks=chunks,
        system_prompt="",
        agent_instruction=agent_instruction,
    )

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": base_system,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=_get_temperature(),

        max_tokens=_get_max_tokens(),
    )

    content = (
        response.choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "A OpenAI não retornou conteúdo."
        )

    return content.strip()


# ============================================================
# MODO DEMONSTRAÇÃO
# ============================================================

def _demo_response():

    return """
### 🤖 Modo demonstração

O pipeline **Retriever → Reranker → AI Service**
foi executado corretamente.

Nenhum provedor de LLM está configurado neste ambiente.

Para ativar a geração de respostas, configure:

`LLM_PROVIDER=gemini`

e:

`GEMINI_API_KEY`

ou:

`LLM_PROVIDER=openai`

e:

`OPENAI_API_KEY`

As evidências recuperadas e as citações continuam
disponíveis para inspeção.
""".strip()


# ============================================================
# GERAÇÃO PRINCIPAL
# ============================================================

def generate_answer(
    query,
    chunks,
    system_prompt=None,
    agent_instruction=None,
):
    """
    Interface principal utilizada pelo RAG.

    Mantém compatibilidade:

        generate_answer(query, chunks)

    E permite futuramente:

        generate_answer(
            query,
            chunks,
            agent_instruction="..."
        )
    """

    query = (
        query or ""
    ).strip()

    if not query:

        return (
            "Não foi fornecida uma pergunta."
        )

    chunks = chunks or []

    provider = _get_provider()

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    if provider == "demo":

        return _demo_response()

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    if provider == "gemini":

        return _generate_gemini(
            query=query,
            chunks=chunks,
            system_prompt=system_prompt,
            agent_instruction=agent_instruction,
        )

    # --------------------------------------------------------
    # OPENAI
    # --------------------------------------------------------

    if provider == "openai":

        return _generate_openai(
            query=query,
            chunks=chunks,
            system_prompt=system_prompt,
            agent_instruction=agent_instruction,
        )

    # --------------------------------------------------------
    # PROVIDER DESCONHECIDO
    # --------------------------------------------------------

    raise RuntimeError(
        f"LLM_PROVIDER inválido: {provider}"
    )


# ============================================================
# GERAÇÃO COM METADADOS
# ============================================================

def generate_answer_result(
    query,
    chunks,
    system_prompt=None,
    agent_instruction=None,
):
    """
    Versão estruturada do AI Service.

    Útil para:
        - auditoria;
        - métricas;
        - dashboard;
        - agentes;
        - monitoramento.
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
            "success": True,
            "answer": answer,
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
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

        return {
            "success": False,
            "answer": "",
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            },
        }


# ============================================================
# STATUS DO SERVIÇO
# ============================================================

def ai_status():
    """
    Retorna o estado atual do serviço de IA.

    Utilizado posteriormente no Dashboard.
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

    return {
        "provider": provider,
        "model": model,
        "configured": configured,
        "temperature": _get_temperature(),
        "max_tokens": _get_max_tokens(),
        "timeout": _get_timeout(),
        "status": (
            "connected"
            if configured
            else (
                "demo"
                if provider == "demo"
                else "not_configured"
            )
        ),
    }


# ============================================================
# LIMPEZA DE CACHE
# ============================================================

def clear_ai_cache():
    """
    Limpa os clientes LLM mantidos em memória.

    Deve ser utilizado quando houver troca de:
        - API Key;
        - provider;
        - configuração.
    """

    _get_gemini_client.cache_clear()

    _get_openai_client.cache_clear()
