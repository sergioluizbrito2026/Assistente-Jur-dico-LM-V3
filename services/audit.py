"""
Assistente Jurídico SaaS IA
services/audit.py

Módulo de auditoria — grava eventos reais na tabela audit_logs.

CORREÇÃO (item 2.6): a versão anterior deste arquivo continha uma
cópia acidental (e quebrada — chamava get_connection()/verify_password()
sem importar nenhum dos dois) de uma versão antiga de services/auth.py.
A função audit() real só fazia print() no console; a tabela audit_logs
nunca era escrita. Esta versão remove todo o código de autenticação
duplicado (a versão correta e única de auth já existe em services/auth.py)
e implementa a gravação de verdade.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from db import get_connection


logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ============================================================
# REGISTRO DE EVENTO
# ============================================================

def audit(
    action: str,
    details: Any = None,
    user_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
) -> bool:
    """
    Registra um evento de auditoria na tabela audit_logs.

    Parâmetros:
        action           -> obrigatório. Ex.: "login", "document_upload",
                             "document_delete", "case_create", "ai_query".
        details           -> texto livre ou dict/list (serializado como
                             JSON na coluna metadata).
        user_id           -> se omitido, tenta obter da sessão atual
                             (services.auth.get_current_user_id()).
        organization_id   -> se omitido, tenta obter da sessão atual
                             (services.auth.get_current_organization_id()).
        entity_type       -> ex.: "document", "case". Opcional.
        entity_id         -> ID da entidade afetada. Opcional.

    Retorna:
        True  -> evento gravado com sucesso.
        False -> falhou (ação vazia ou erro de banco) — nunca lança
                 exceção, para não derrubar o fluxo principal do app
                 por causa de um log de auditoria.
    """

    action = (action or "").strip()

    if not action:
        logger.warning("audit() chamado sem 'action' — evento ignorado.")
        return False

    # --------------------------------------------------------
    # Preenche user_id/organization_id a partir da sessão atual
    # quando não informados explicitamente.
    # --------------------------------------------------------

    if user_id is None or organization_id is None:

        try:
            from services.auth import (
                get_current_user_id,
                get_current_organization_id,
            )

            if user_id is None:
                user_id = get_current_user_id()

            if organization_id is None:
                organization_id = get_current_organization_id()

        except Exception:
            # Contexto sem sessão Streamlit ativa (ex.: script standalone).
            # Segue sem user_id/organization_id — a tabela permite NULL.
            pass

    # --------------------------------------------------------
    # Normaliza metadata
    # --------------------------------------------------------

    if isinstance(details, (dict, list)):

        try:
            metadata = json.dumps(details, ensure_ascii=False, default=str)
        except Exception:
            metadata = str(details)

    elif details is None:
        metadata = None

    else:
        metadata = json.dumps(
            {"message": str(details)},
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Gravação
    # --------------------------------------------------------

    try:

        with get_connection() as c:

            c.execute(
                """
                INSERT INTO audit_logs(
                    organization_id,
                    user_id,
                    action,
                    entity_type,
                    entity_id,
                    metadata,
                    created_at
                )
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    organization_id,
                    user_id,
                    action,
                    entity_type,
                    entity_id,
                    metadata,
                    _now(),
                ),
            )

        return True

    except Exception as exc:

        logger.exception(
            "Falha ao gravar log de auditoria (action=%s): %s",
            action,
            exc,
        )

        return False


# ============================================================
# CONSULTA
# ============================================================

def list_audit_logs(
    organization_id: int,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Lista os eventos de auditoria mais recentes de uma organização,
    do mais novo para o mais antigo.

    Útil para alimentar a tela "Auditoria" do app com dado real
    em vez de conteúdo estático.
    """

    try:
        organization_id = int(organization_id)
    except (TypeError, ValueError):
        return []

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 200

    limit = max(1, min(limit, 1000))

    with get_connection() as c:

        rows = c.execute(
            """
            SELECT *
            FROM audit_logs
            WHERE organization_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                organization_id,
                limit,
            ),
        ).fetchall()

    results: List[Dict[str, Any]] = []

    for row in rows:

        item = dict(row)

        if item.get("metadata"):

            try:
                item["metadata"] = json.loads(item["metadata"])
            except (TypeError, ValueError):
                pass  # mantém como string se não for JSON válido

        results.append(item)

    return results


def count_audit_logs(organization_id: int) -> int:
    """
    Retorna o total de eventos de auditoria de uma organização.
    """

    try:
        organization_id = int(organization_id)
    except (TypeError, ValueError):
        return 0

    with get_connection() as c:

        row = c.execute(
            """
            SELECT COUNT(*) AS total
            FROM audit_logs
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    return int(row["total"]) if row else 0


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural. Não grava nada no banco.
    """

    required = [
        "audit",
        "list_audit_logs",
        "count_audit_logs",
    ]

    missing = [
        name
        for name in required
        if name not in globals()
    ]

    return {
        "module": "services.audit",
        "status": "ok" if not missing else "error",
        "required_functions": required,
        "missing_functions": missing,
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print("AUDIT.PY - SELF TEST")
    print("=" * 60)

    print(f"Status: {result['status']}")
    print(f"Funções obrigatórias: {len(result['required_functions'])}")
    print(f"Funções ausentes: {result['missing_functions']}")

    print("=" * 60)
