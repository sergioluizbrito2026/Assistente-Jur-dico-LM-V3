"""
Assistente Jurídico SaaS IA V3
services/documents.py

Serviço de gerenciamento de documentos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from db import get_connection


def _normalize_org_id(org_id: Any) -> int:
    try:
        org_id = int(org_id)
    except (TypeError, ValueError):
        raise ValueError("org_id inválido.")
    if org_id <= 0:
        raise ValueError("org_id deve ser maior que zero.")
    return org_id


def _normalize_document_id(document_id: Any) -> int:
    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        raise ValueError("document_id inválido.")
    if document_id <= 0:
        raise ValueError("document_id deve ser maior que zero.")
    return document_id


def list_documents(org_id: int) -> List[Dict[str, Any]]:
    org_id = _normalize_org_id(org_id)
    with get_connection() as c:
        rows = c.execute(
            "SELECT * FROM documents WHERE organization_id = ? ORDER BY id DESC",
            (org_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(document_id: int, org_id: int) -> Optional[Dict[str, Any]]:
    document_id = _normalize_document_id(document_id)
    org_id = _normalize_org_id(org_id)
    with get_connection() as c:
        row = c.execute(
            "SELECT * FROM documents WHERE id = ? AND organization_id = ? LIMIT 1",
            (document_id, org_id),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def document_exists(document_id: int, org_id: int) -> bool:
    return get_document(document_id=document_id, org_id=org_id) is not None


def create_document(org_id: int, name: str, **kwargs: Any) -> int:
    org_id = _normalize_org_id(org_id)
    name = (name or "").strip()
    if not name:
        raise ValueError("O nome do documento é obrigatório.")

    with get_connection() as c:
        columns_rows = c.execute("PRAGMA table_info(documents)").fetchall()
        columns = {row["name"] for row in columns_rows}

        data: Dict[str, Any] = {"organization_id": org_id, "name": name}

        optional_fields = [
            "filename", "file_name", "path", "file_path", "mime_type",
            "size", "file_size", "status", "created_at", "updated_at",
        ]
        for field in optional_fields:
            if field in columns and field in kwargs:
                data[field] = kwargs[field]

        valid_data = {key: value for key, value in data.items() if key in columns}

        if "organization_id" not in valid_data:
            raise RuntimeError("A tabela documents não possui organization_id.")
        if "name" not in valid_data:
            raise RuntimeError("A tabela documents não possui a coluna name.")

        fields = list(valid_data.keys())
        placeholders = ", ".join("?" for _ in fields)
        sql = f"INSERT INTO documents ({', '.join(fields)}) VALUES ({placeholders})"
        cursor = c.execute(sql, [valid_data[field] for field in fields])
        return int(cursor.lastrowid)


# ============================================================
# EXCLUSÃO
# ============================================================
#
# CORREÇÃO (item 2.7): antes, excluir um documento removia a
# linha em `documents` (com CASCADE para `chunks`), mas nunca
# atualizava o índice vetorial FAISS. Os vetores do documento
# excluído continuavam no índice, e o RAG podia continuar
# recuperando e citando um documento que a UI já dizia não
# existir mais.
#
# Agora, após um delete bem-sucedido, o índice da organização
# é reconstruído a partir do que sobrou no SQLite — que já não
# tem mais os chunks do documento excluído (graças ao ON DELETE
# CASCADE). rebuild_index() é O(n) sobre os chunks restantes;
# para volumes grandes, o ideal futuro é remoção seletiva de
# vetores em vez de reconstrução completa — ver observação no
# roadmap (services/embeddings.py não expõe isso ainda).

def delete_document(document_id: int, org_id: int) -> bool:
    """
    Exclui um documento pertencente à organização e mantém o
    índice vetorial FAISS consistente com o SQLite.

    Retorna:
        True  -> documento excluído (e índice reconstruído).
        False -> documento não encontrado.
    """

    document_id = _normalize_document_id(document_id)
    org_id = _normalize_org_id(org_id)

    with get_connection() as c:
        cursor = c.execute(
            "DELETE FROM documents WHERE id = ? AND organization_id = ?",
            (document_id, org_id),
        )
        deleted = cursor.rowcount > 0

    if deleted:
        try:
            from services.embeddings import rebuild_index
            rebuild_index(org_id)
        except Exception:
            # Não derruba a exclusão do documento por causa de uma
            # falha na reconstrução do índice — mas registra, porque
            # deixa o RAG temporariamente inconsistente até o próximo
            # rebuild manual ou upsert.
            import logging
            logging.getLogger(__name__).exception(
                "Documento %s excluído, mas falhou ao reconstruir o "
                "índice FAISS da organização %s.", document_id, org_id,
            )

    return deleted


def count_documents(org_id: int) -> int:
    org_id = _normalize_org_id(org_id)
    with get_connection() as c:
        row = c.execute(
            "SELECT COUNT(*) AS total FROM documents WHERE organization_id = ?",
            (org_id,),
        ).fetchone()
    if row is None:
        return 0
    return int(row["total"])


def get_document_with_chunks(document_id: int, org_id: int) -> Optional[Dict[str, Any]]:
    document = get_document(document_id=document_id, org_id=org_id)
    if document is None:
        return None
    with get_connection() as c:
        rows = c.execute(
            """
            SELECT * FROM chunks
            WHERE document_id = ? AND organization_id = ?
            ORDER BY chunk_index ASC, id ASC
            """,
            (int(document_id), int(org_id)),
        ).fetchall()
    result = dict(document)
    result["chunks"] = [dict(row) for row in rows]
    result["chunk_count"] = len(result["chunks"])
    return result


def document_status(org_id: int) -> Dict[str, Any]:
    org_id = _normalize_org_id(org_id)
    total_documents = count_documents(org_id)
    with get_connection() as c:
        row = c.execute(
            "SELECT COUNT(*) AS total FROM chunks WHERE organization_id = ?",
            (org_id,),
        ).fetchone()
    total_chunks = int(row["total"]) if row else 0
    return {
        "organization_id": org_id,
        "documents": total_documents,
        "chunks": total_chunks,
        "status": "ready",
    }


def self_test() -> Dict[str, Any]:
    required = [
        "list_documents", "get_document", "document_exists", "create_document",
        "delete_document", "count_documents", "get_document_with_chunks",
        "document_status",
    ]
    missing = [name for name in required if name not in globals()]
    return {
        "module": "services.documents",
        "status": "ok" if not missing else "error",
        "required_functions": required,
        "missing_functions": missing,
    }


if __name__ == "__main__":
    result = self_test()
    print("=" * 60)
    print("DOCUMENTS.PY - SELF TEST")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Funções ausentes: {result['missing_functions']}")
    print("=" * 60)
