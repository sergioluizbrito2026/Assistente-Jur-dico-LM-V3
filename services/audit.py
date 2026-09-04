"""
Assistente Jurídico SaaS IA V3
services/audit.py

Sistema de auditoria da plataforma.

Responsabilidades:
- Registrar ações dos usuários.
- Registrar eventos de documentos, casos e IA.
- Manter organização e usuário associados ao evento.
- Serializar metadata com segurança.
- Evitar que problemas de auditoria derrubem a aplicação.
- Fornecer funções auxiliares para consulta dos logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from db import get_connection


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MAX_METADATA_CHARS = 10000
MAX_ERROR_CHARS = 500


# ============================================================
# UTILITÁRIOS
# ============================================================

def _safe_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:
    """
    Converte um valor para inteiro com segurança.
    """

    try:
        if value is None:
            return default

        return int(value)

    except (TypeError, ValueError):
        return default


def _safe_text(
    value: Any,
    default: str = "",
    max_length: Optional[int] = None,
) -> str:
    """
    Converte qualquer valor para texto.
    """

    if value is None:
        text = default

    else:
        try:
            text = str(value).strip()
        except Exception:
            text = default

    if max_length is not None:
        text = text[:max_length]

    return text


def _serialize_metadata(
    metadata: Any = None,
) -> str:
    """
    Serializa metadata para JSON.

    Nunca permite que um objeto não serializável
    interrompa o registro de auditoria.
    """

    if metadata is None:
        metadata = {}

    if not isinstance(metadata, dict):
        metadata = {
            "value": metadata
        }

    try:
        result = json.dumps(
            metadata,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    except Exception:
        result = json.dumps(
            {
                "serialization_error": True
            },
            ensure_ascii=False,
        )

    return result[:MAX_METADATA_CHARS]


def _extract_user_data(
    user: Any,
) -> Dict[str, Optional[int]]:
    """
    Extrai organization_id e user_id de diferentes
    formatos de usuário.

    Compatível com:
        dict
        objetos
        None
    """

    if user is None:
        return {
            "organization_id": None,
            "user_id": None,
        }

    organization_id = None
    user_id = None

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(user, dict):

        organization_id = (
            user.get("organization_id")
            or user.get("org_id")
        )

        user_id = (
            user.get("id")
            or user.get("user_id")
        )

    # --------------------------------------------------------
    # Objeto
    # --------------------------------------------------------

    else:

        try:
            organization_id = (
                getattr(
                    user,
                    "organization_id",
                    None,
                )
                or getattr(
                    user,
                    "org_id",
                    None,
                )
            )
        except Exception:
            organization_id = None

        try:
            user_id = (
                getattr(
                    user,
                    "id",
                    None,
                )
                or getattr(
                    user,
                    "user_id",
                    None,
                )
            )
        except Exception:
            user_id = None

    return {
        "organization_id": _safe_int(
            organization_id
        ),
        "user_id": _safe_int(
            user_id
        ),
    }


# ============================================================
# REGISTRO DE AUDITORIA
# ============================================================

def audit(
    user: Any,
    action: str,
    entity_type: str,
    entity_id: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Registra uma ação na tabela audit_logs.

    Retorna:

        True  -> registro realizado
        False -> falha controlada

    A auditoria não deve derrubar a aplicação principal.
    """

    user_data = _extract_user_data(user)

    organization_id = user_data[
        "organization_id"
    ]

    user_id = user_data[
        "user_id"
    ]

    action = _safe_text(
        action,
        default="unknown_action",
        max_length=200,
    )

    entity_type = _safe_text(
        entity_type,
        default="unknown",
        max_length=100,
    )

    entity_id = _safe_int(
        entity_id
    )

    metadata_json = _serialize_metadata(
        metadata
    )

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

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
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    organization_id,
                    user_id,
                    action,
                    entity_type,
                    entity_id,
                    metadata_json,
                    created_at,
                ),
            )

        return True

    except Exception:
        # Auditoria não deve interromper:
        # login, upload, RAG, IA ou operações do usuário.
        return False


# ============================================================
# EVENTOS DE AUDITORIA
# ============================================================

def audit_event(
    user: Any,
    action: str,
    entity_type: str,
    entity_id: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Alias semântico para audit().
    """

    return audit(
        user=user,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        metadata=metadata,
    )


def audit_ai(
    user: Any,
    action: str,
    metadata: Optional[Dict[str, Any]] = None,
    entity_id: Any = None,
) -> bool:
    """
    Registra eventos relacionados à IA.

    Exemplos:

        audit_ai(
            user,
            "rag_query",
            {
                "query": "...",
                "model": "...",
            }
        )
    """

    return audit(
        user=user,
        action=action,
        entity_type="ai",
        entity_id=entity_id,
        metadata=metadata,
    )


def audit_document(
    user: Any,
    action: str,
    document_id: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Registra eventos relacionados a documentos.
    """

    return audit(
        user=user,
        action=action,
        entity_type="document",
        entity_id=document_id,
        metadata=metadata,
    )


def audit_case(
    user: Any,
    action: str,
    case_id: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Registra eventos relacionados a processos/casos.
    """

    return audit(
        user=user,
        action=action,
        entity_type="case",
        entity_id=case_id,
        metadata=metadata,
    )


# ============================================================
# CONSULTA DE AUDITORIA
# ============================================================

def list_audit_logs(
    org_id: Any,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retorna os registros de auditoria de uma organização.
    """

    org_id = _safe_int(org_id)

    if org_id is None:
        return []

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100

    limit = max(
        1,
        min(limit, 1000),
    )

    try:

        with get_connection() as c:

            rows = c.execute(
                """
                SELECT
                    id,
                    organization_id,
                    user_id,
                    action,
                    entity_type,
                    entity_id,
                    metadata,
                    created_at
                FROM audit_logs
                WHERE organization_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    org_id,
                    limit,
                ),
            ).fetchall()

        results = []

        for row in rows:

            item = dict(row)

            # ------------------------------------------------
            # Tenta converter metadata novamente para dict
            # ------------------------------------------------

            raw_metadata = item.get(
                "metadata"
            )

            try:

                item["metadata"] = json.loads(
                    raw_metadata
                ) if raw_metadata else {}

            except Exception:

                item["metadata"] = {
                    "raw": _safe_text(
                        raw_metadata
                    )
                }

            results.append(
                item
            )

        return results

    except Exception:
        return []


# ============================================================
# LIMPEZA DE LOGS
# ============================================================

def delete_old_audit_logs(
    org_id: Any,
    before_date: str,
) -> int:
    """
    Remove logs anteriores a uma determinada data.

    Útil para retenção de dados em planos SaaS.
    """

    org_id = _safe_int(
        org_id
    )

    before_date = _safe_text(
        before_date
    )

    if org_id is None or not before_date:
        return 0

    try:

        with get_connection() as c:

            cursor = c.execute(
                """
                DELETE FROM audit_logs
                WHERE organization_id = ?
                  AND created_at < ?
                """,
                (
                    org_id,
                    before_date,
                ),
            )

            return max(
                0,
                int(
                    cursor.rowcount
                ),
            )

    except Exception:
        return 0


# ============================================================
# HEALTH CHECK
# ============================================================

def audit_status() -> Dict[str, Any]:
    """
    Verifica se o sistema de auditoria está operacional.
    """

    try:

        with get_connection() as c:

            c.execute(
                """
                SELECT 1
                FROM audit_logs
                LIMIT 1
                """
            ).fetchone()

        return {
            "configured": True,
            "status": "ready",
            "table": "audit_logs",
        }

    except Exception as exc:

        return {
            "configured": False,
            "status": "error",
            "table": "audit_logs",
            "error": (
                f"{type(exc).__name__}: "
                f"{str(exc)[:MAX_ERROR_CHARS]}"
            ),
        }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural do módulo.

    Não cria registros no banco.
    """

    required = [
        "audit",
        "audit_event",
        "audit_ai",
        "audit_document",
        "audit_case",
        "list_audit_logs",
        "delete_old_audit_logs",
        "audit_status",
    ]

    missing = [
        name
        for name in required
        if not callable(
            globals().get(name)
        )
    ]

    return {
        "module": "services.audit",
        "status": (
            "ok"
            if not missing
            else "error"
        ),
        "required_functions": required,
        "missing_functions": missing,
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print("AUDIT.PY V3 - SELF TEST")
    print("=" * 60)

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Funções obrigatórias: "
        f"{len(result['required_functions'])}"
    )

    print(
        f"Funções ausentes: "
        f"{result['missing_functions']}"
    )

    print(
        f"Audit status: "
        f"{audit_status()['status']}"
    )

    print("=" * 60)
