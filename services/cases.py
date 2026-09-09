"""
Assistente Jurídico SaaS IA V3
services/cases.py

Serviço responsável pelo gerenciamento de casos jurídicos.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from db import get_connection


CASE_STATUSES = ["Aberto", "Em andamento", "Aguardando", "Concluído", "Arquivado"]
CASE_PRIORITIES = ["Baixa", "Média", "Alta", "Crítica"]

# CORREÇÃO (item secundário — bug de status): o schema (db.py) define
# DEFAULT 'Em andamento' para a coluna status. create_case() não
# incluía "status" no INSERT, então o banco aplicava esse default,
# mas a função retornava "status": "Aberto" pro chamador — os dois
# nunca batiam. Agora usamos a mesma constante nos dois lugares.
DEFAULT_CASE_STATUS = "Em andamento"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_org_id(org_id: Any) -> int:
    try:
        value = int(org_id)
    except (TypeError, ValueError):
        raise ValueError("Organização inválida.")
    if value <= 0:
        raise ValueError("Organização inválida.")
    return value


def _safe_case_id(case_id: Any) -> int:
    try:
        value = int(case_id)
    except (TypeError, ValueError):
        raise ValueError("ID do caso inválido.")
    if value <= 0:
        raise ValueError("ID do caso inválido.")
    return value


def _clean_text(value: Any, field_name: str, required: bool = True) -> str:
    value = str(value or "").strip()
    if required and not value:
        raise ValueError(f"O campo '{field_name}' é obrigatório.")
    return value


def _row_to_dict(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def create_case(org, title, client, category, priority):
    org_id = _safe_org_id(org)
    title = _clean_text(title, "title")
    client = _clean_text(client, "client")
    category = _clean_text(category, "category")
    priority = _clean_text(priority, "priority")

    if priority not in CASE_PRIORITIES:
        raise ValueError(f"Prioridade inválida. Utilize: {', '.join(CASE_PRIORITIES)}.")

    created_at = _now()

    with get_connection() as c:
        cursor = c.execute(
            """
            INSERT INTO cases(organization_id, title, client, category, priority, status, created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (org_id, title, client, category, priority, DEFAULT_CASE_STATUS, created_at),
        )
        case_id = cursor.lastrowid

    return {
        "success": True,
        "case_id": case_id,
        "organization_id": org_id,
        "title": title,
        "client": client,
        "category": category,
        "priority": priority,
        "created_at": created_at,
        "status": DEFAULT_CASE_STATUS,
    }


def list_cases(org, limit: int = 100) -> List[Dict[str, Any]]:
    org_id = _safe_org_id(org)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 1000))

    with get_connection() as c:
        rows = c.execute(
            "SELECT * FROM cases WHERE organization_id = ? ORDER BY id DESC LIMIT ?",
            (org_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_case(org, case_id) -> Optional[Dict[str, Any]]:
    org_id = _safe_org_id(org)
    case_id = _safe_case_id(case_id)
    with get_connection() as c:
        row = c.execute(
            "SELECT * FROM cases WHERE id = ? AND organization_id = ? LIMIT 1",
            (case_id, org_id),
        ).fetchone()
    return _row_to_dict(row)


def update_case(org, case_id, title=None, client=None, category=None, priority=None, status=None):
    org_id = _safe_org_id(org)
    case_id = _safe_case_id(case_id)

    fields = []
    values = []

    if title is not None:
        fields.append("title = ?")
        values.append(_clean_text(title, "title"))

    if client is not None:
        fields.append("client = ?")
        values.append(_clean_text(client, "client"))

    if category is not None:
        fields.append("category = ?")
        values.append(_clean_text(category, "category"))

    if priority is not None:
        priority = _clean_text(priority, "priority")
        if priority not in CASE_PRIORITIES:
            raise ValueError("Prioridade inválida.")
        fields.append("priority = ?")
        values.append(priority)

    if status is not None:
        status = _clean_text(status, "status")
        if status not in CASE_STATUSES:
            raise ValueError(f"Status inválido. Utilize: {', '.join(CASE_STATUSES)}.")
        fields.append("status = ?")
        values.append(status)

    if not fields:
        return {
            "success": False, "updated": False, "case_id": case_id,
            "message": "Nenhum campo informado para atualização.",
        }

    values.extend([case_id, org_id])

    with get_connection() as c:
        cursor = c.execute(
            f"UPDATE cases SET {', '.join(fields)} WHERE id = ? AND organization_id = ?",
            tuple(values),
        )
        updated = cursor.rowcount > 0

    return {"success": updated, "updated": updated, "case_id": case_id, "organization_id": org_id}


def update_case_status(org, case_id, status):
    return update_case(org=org, case_id=case_id, status=status)


def delete_case(org, case_id):
    org_id = _safe_org_id(org)
    case_id = _safe_case_id(case_id)
    with get_connection() as c:
        cursor = c.execute(
            "DELETE FROM cases WHERE id = ? AND organization_id = ?",
            (case_id, org_id),
        )
        deleted = cursor.rowcount > 0
    return {"success": deleted, "deleted": deleted, "case_id": case_id, "organization_id": org_id}


def search_cases(org, query, limit: int = 100) -> List[Dict[str, Any]]:
    org_id = _safe_org_id(org)
    query = _clean_text(query, "query")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 1000))
    pattern = f"%{query}%"

    with get_connection() as c:
        rows = c.execute(
            """
            SELECT * FROM cases
            WHERE organization_id = ?
              AND (title LIKE ? OR client LIKE ? OR category LIKE ?)
            ORDER BY id DESC LIMIT ?
            """,
            (org_id, pattern, pattern, pattern, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def list_cases_by_priority(org, priority, limit: int = 100):
    org_id = _safe_org_id(org)
    priority = _clean_text(priority, "priority")
    if priority not in CASE_PRIORITIES:
        raise ValueError("Prioridade inválida.")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 1000))

    with get_connection() as c:
        rows = c.execute(
            "SELECT * FROM cases WHERE organization_id = ? AND priority = ? ORDER BY id DESC LIMIT ?",
            (org_id, priority, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def count_cases(org):
    org_id = _safe_org_id(org)
    with get_connection() as c:
        row = c.execute(
            "SELECT COUNT(*) AS total FROM cases WHERE organization_id = ?",
            (org_id,),
        ).fetchone()
    return int(row["total"] if row else 0)


def count_cases_by_priority(org):
    org_id = _safe_org_id(org)
    result = {priority: 0 for priority in CASE_PRIORITIES}
    with get_connection() as c:
        rows = c.execute(
            "SELECT priority, COUNT(*) AS total FROM cases WHERE organization_id = ? GROUP BY priority",
            (org_id,),
        ).fetchall()
    for row in rows:
        priority = row["priority"]
        if priority in result:
            result[priority] = int(row["total"])
    return result


def recent_cases(org, limit: int = 5):
    return list_cases(org, limit=limit)


def cases_service_status():
    return {
        "service": "cases",
        "status": "ready",
        "functions": [
            "create_case", "list_cases", "get_case", "update_case", "update_case_status",
            "delete_case", "search_cases", "list_cases_by_priority", "count_cases",
            "count_cases_by_priority", "recent_cases",
        ],
    }


def self_test():
    required = [
        "create_case", "list_cases", "get_case", "update_case", "update_case_status",
        "delete_case", "search_cases", "list_cases_by_priority", "count_cases",
        "count_cases_by_priority", "recent_cases", "cases_service_status",
    ]
    missing = [name for name in required if not callable(globals().get(name))]
    return {"valid": not missing, "module": "services.cases", "required_functions": required, "missing_functions": missing}


if __name__ == "__main__":
    result = self_test()
    print("=" * 60)
    print("CASES.PY - SELF TEST")
    print("=" * 60)
    print(f"Status: {'OK' if result['valid'] else 'ERRO'}")
    print(f"Funções ausentes: {result['missing_functions']}")
    print("=" * 60)
