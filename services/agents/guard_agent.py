"""
Guard Agent
Camada de segurança do Assistente Jurídico IA V3.

Responsabilidades:
- Detectar prompt injection.
- Bloquear tentativas de ignorar regras do sistema.
- Bloquear solicitações para inventar informações.
- Identificar solicitações fora do escopo jurídico.
- Proteger o contexto RAG.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


AGENT_NAME = "guard_agent"
AGENT_LABEL = "Agente de Segurança"


# Padrões básicos de prompt injection.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"esqueça\s+(todas\s+as\s+)?instruções",
    r"ignore\s+(todas\s+as\s+)?instruções",
    r"desconsidere\s+(todas\s+as\s+)?instruções",
    r"revele\s+(seu\s+)?prompt",
    r"mostre\s+(seu\s+)?system\s+prompt",
    r"qual\s+é\s+seu\s+system\s+prompt",
    r"reveal\s+your\s+system\s+prompt",
    r"show\s+me\s+your\s+system\s+prompt",
    r"bypass\s+(the\s+)?security",
    r"bypass\s+(as\s+)?regras",
    r"ignore\s+security",
    r"jailbreak",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"aja\s+como\s+se\s+não\s+tivesse\s+restrições",
]


# Solicitações que tentam induzir a IA a fabricar conteúdo.
FABRICATION_PATTERNS = [
    r"invente\s+(uma\s+)?lei",
    r"invente\s+(uma\s+)?jurisprudência",
    r"invente\s+(um\s+)?processo",
    r"crie\s+uma\s+jurisprudência\s+fictícia",
    r"crie\s+uma\s+lei\s+fictícia",
    r"fabrique\s+(uma\s+)?jurisprudência",
    r"falsifique\s+(uma\s+)?informação",
    r"make\s+up\s+(a\s+)?case",
    r"fabricate\s+(a\s+)?legal",
]


# Pedidos que tentam obter informações internas do sistema.
SYSTEM_ACCESS_PATTERNS = [
    r"mostre\s+as\s+instruções\s+internas",
    r"mostre\s+as\s+regras\s+internas",
    r"mostre\s+o\s+prompt\s+interno",
    r"exiba\s+o\s+prompt\s+do\s+sistema",
    r"liste\s+as\s+variáveis\s+de\s+ambiente",
    r"mostre\s+as\s+chaves\s+da\s+api",
    r"mostre\s+a\s+api\s*key",
    r"show\s+environment\s+variables",
    r"show\s+api\s+keys",
    r"show\s+the\s+internal\s+instructions",
]


def _normalize(text: Any) -> str:
    """
    Normaliza texto para análise.
    """
    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def _matches_patterns(
    text: str,
    patterns: List[str],
) -> List[str]:
    """
    Retorna os padrões encontrados.
    """
    matches: List[str] = []

    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.append(pattern)
        except re.error:
            continue

    return matches


def inspect(query: str) -> Dict[str, Any]:
    """
    Analisa uma solicitação antes da execução do agente.

    Retorna:
        allowed: se a solicitação pode continuar.
        risk_level: low / medium / high.
        reasons: motivos encontrados.
        categories: categorias detectadas.
    """

    text = _normalize(query)

    if not text:
        return {
            "allowed": False,
            "risk_level": "medium",
            "reasons": ["Solicitação vazia."],
            "categories": ["invalid_input"],
        }

    injection_matches = _matches_patterns(
        text,
        INJECTION_PATTERNS,
    )

    fabrication_matches = _matches_patterns(
        text,
        FABRICATION_PATTERNS,
    )

    system_matches = _matches_patterns(
        text,
        SYSTEM_ACCESS_PATTERNS,
    )

    categories: List[str] = []
    reasons: List[str] = []

    if injection_matches:
        categories.append("prompt_injection")
        reasons.append(
            "Foi detectada uma tentativa de alterar ou ignorar "
            "as instruções de segurança."
        )

    if fabrication_matches:
        categories.append("fabrication")
        reasons.append(
            "A solicitação tenta induzir a geração de "
            "informações jurídicas não verificadas."
        )

    if system_matches:
        categories.append("system_access")
        reasons.append(
            "A solicitação tenta acessar informações internas "
            "ou credenciais do sistema."
        )

    if categories:
        return {
            "allowed": False,
            "risk_level": "high",
            "reasons": reasons,
            "categories": categories,
        }

    return {
        "allowed": True,
        "risk_level": "low",
        "reasons": [],
        "categories": [],
    }


def guard(query: str) -> Dict[str, Any]:
    """
    Alias público para inspect().
    """
    return inspect(query)


def run(query: str) -> Dict[str, Any]:
    """
    Interface padronizada do Guard Agent.
    """

    result = inspect(query)

    if result["allowed"]:
        message = "Solicitação aprovada para processamento."

    else:
        message = (
            "A solicitação foi bloqueada pela camada de segurança. "
            "O sistema não pode ignorar suas regras, expor instruções "
            "internas ou inventar informações jurídicas."
        )

    return {
        "success": result["allowed"],
        "agent": AGENT_NAME,
        "label": AGENT_LABEL,
        "allowed": result["allowed"],
        "risk_level": result["risk_level"],
        "categories": result["categories"],
        "reasons": result["reasons"],
        "message": message,
        "query": query,
    }


def status() -> Dict[str, Any]:
    """
    Status do Guard Agent.
    """

    return {
        "agent": AGENT_NAME,
        "label": AGENT_LABEL,
        "status": "available",
        "prompt_injection_detection": True,
        "fabrication_detection": True,
        "system_access_detection": True,
    }
