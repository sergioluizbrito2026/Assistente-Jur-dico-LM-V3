"""
Assistente Jurídico SaaS IA V3
services/documents.py

Serviço de gerenciamento de documentos.

Responsabilidades:
- Listar documentos da organização.
- Buscar documento por ID.
- Criar documento.
- Excluir documento.
- Verificar existência.
- Contar documentos.
- Manter isolamento por organização.
- Preparar integração com ingestion.py e embeddings.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from db import get_connection


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def _normalize_org_id(org_id: Any) -> int:
    """
    Normaliza e valida o ID da organização.
    """

    try:
        org_id = int(org_id)
    except (TypeError, ValueError):
        raise ValueError("org_id inválido.")

    if org_id <= 0:
        raise ValueError("org_id deve ser maior que zero.")

    return org_id


def _normalize_document_id(document_id: Any) -> int:
    """
    Normaliza e valida o ID do documento.
    """

    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        raise ValueError("document_id inválido.")

    if document_id <= 0:
        raise ValueError(
            "document_id deve ser maior que zero."
        )

    return document_id


# ============================================================
# LISTAGEM
# ============================================================

def list_documents(
    org_id: int,
) -> List[Dict[str, Any]]:
    """
    Lista todos os documentos pertencentes à organização.

    Os documentos são retornados do mais recente
    para o mais antigo.
    """

    org_id = _normalize_org_id(org_id)

    with get_connection() as c:

        rows = c.execute(
            """
            SELECT *
            FROM documents
            WHERE organization_id = ?
            ORDER BY id DESC
            """,
            (org_id,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# BUSCAR DOCUMENTO
# ============================================================

def get_document(
    document_id: int,
    org_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Retorna um documento específico.

    O organization_id é obrigatório para impedir
    acesso cruzado entre organizações.
    """

    document_id = _normalize_document_id(
        document_id
    )

    org_id = _normalize_org_id(
        org_id
    )

    with get_connection() as c:

        row = c.execute(
            """
            SELECT *
            FROM documents
            WHERE
                id = ?
                AND organization_id = ?
            LIMIT 1
            """,
            (
                document_id,
                org_id,
            ),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# ============================================================
# EXISTÊNCIA
# ============================================================

def document_exists(
    document_id: int,
    org_id: int,
) -> bool:
    """
    Verifica se o documento pertence à organização.
    """

    return (
        get_document(
            document_id=document_id,
            org_id=org_id,
        )
        is not None
    )


# ============================================================
# CRIAÇÃO
# ============================================================

def create_document(
    org_id: int,
    name: str,
    **kwargs: Any,
) -> int:
    """
    Cria um registro de documento.

    A função tenta utilizar os campos mais comuns
    encontrados no schema V3.

    Retorna o ID criado.

    Observação:
    A ingestão do arquivo/chunks deve ser realizada
    posteriormente pelo ingestion.py.
    """

    org_id = _normalize_org_id(
        org_id
    )

    name = (
        name or ""
    ).strip()

    if not name:
        raise ValueError(
            "O nome do documento é obrigatório."
        )

    with get_connection() as c:

        # ----------------------------------------------------
        # Descobre as colunas existentes
        # ----------------------------------------------------

        columns_rows = c.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()

        columns = {
            row["name"]
            for row in columns_rows
        }

        data: Dict[str, Any] = {
            "organization_id": org_id,
            "name": name,
        }

        # Campos opcionais
        optional_fields = [
            "filename",
            "file_name",
            "path",
            "file_path",
            "mime_type",
            "size",
            "file_size",
            "status",
            "created_at",
            "updated_at",
        ]

        for field in optional_fields:

            if field in columns and field in kwargs:

                data[field] = kwargs[field]

        valid_data = {
            key: value
            for key, value in data.items()
            if key in columns
        }

        if "organization_id" not in valid_data:
            raise RuntimeError(
                "A tabela documents não possui "
                "organization_id."
            )

        if "name" not in valid_data:
            raise RuntimeError(
                "A tabela documents não possui "
                "a coluna name."
            )

        fields = list(
            valid_data.keys()
        )

        placeholders = ", ".join(
            "?"
            for _ in fields
        )

        sql = f"""
            INSERT INTO documents (
                {", ".join(fields)}
            )
            VALUES (
                {placeholders}
            )
        """

        cursor = c.execute(
            sql,
            [
                valid_data[field]
                for field in fields
            ],
        )

        return int(
            cursor.lastrowid
        )


# ============================================================
# EXCLUSÃO
# ============================================================

def delete_document(
    document_id: int,
    org_id: int,
) -> bool:
    """
    Exclui um documento pertencente à organização.

    Retorna:
        True  -> documento excluído
        False -> documento não encontrado
    """

    document_id = _normalize_document_id(
        document_id
    )

    org_id = _normalize_org_id(
        org_id
    )

    with get_connection() as c:

        cursor = c.execute(
            """
            DELETE FROM documents
            WHERE
                id = ?
                AND organization_id = ?
            """,
            (
                document_id,
                org_id,
            ),
        )

        return cursor.rowcount > 0


# ============================================================
# CONTAGEM
# ============================================================

def count_documents(
    org_id: int,
) -> int:
    """
    Retorna a quantidade de documentos da organização.
    """

    org_id = _normalize_org_id(
        org_id
    )

    with get_connection() as c:

        row = c.execute(
            """
            SELECT COUNT(*) AS total
            FROM documents
            WHERE organization_id = ?
            """,
            (org_id,),
        ).fetchone()

    if row is None:
        return 0

    return int(
        row["total"]
    )


# ============================================================
# DOCUMENTOS COM CHUNKS
# ============================================================

def get_document_with_chunks(
    document_id: int,
    org_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Retorna um documento juntamente com seus chunks.

    Útil para:
    - inspeção;
    - auditoria;
    - debugging;
    - visualização no app.
    """

    document = get_document(
        document_id=document_id,
        org_id=org_id,
    )

    if document is None:
        return None

    with get_connection() as c:

        rows = c.execute(
            """
            SELECT *
            FROM chunks
            WHERE
                document_id = ?
                AND organization_id = ?
            ORDER BY chunk_index ASC, id ASC
            """,
            (
                int(document_id),
                int(org_id),
            ),
        ).fetchall()

    result = dict(document)

    result["chunks"] = [
        dict(row)
        for row in rows
    ]

    result["chunk_count"] = len(
        result["chunks"]
    )

    return result


# ============================================================
# STATUS
# ============================================================

def document_status(
    org_id: int,
) -> Dict[str, Any]:
    """
    Retorna informações resumidas sobre os documentos
    da organização.
    """

    org_id = _normalize_org_id(
        org_id
    )

    total_documents = count_documents(
        org_id
    )

    with get_connection() as c:

        row = c.execute(
            """
            SELECT COUNT(*) AS total
            FROM chunks
            WHERE organization_id = ?
            """,
            (org_id,),
        ).fetchone()

    total_chunks = (
        int(row["total"])
        if row
        else 0
    )

    return {
        "organization_id": org_id,
        "documents": total_documents,
        "chunks": total_chunks,
        "status": "ready",
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural do módulo.

    Não cria nem exclui dados.
    """

    required = [
        "list_documents",
        "get_document",
        "document_exists",
        "create_document",
        "delete_document",
        "count_documents",
        "get_document_with_chunks",
        "document_status",
    ]

    missing = [
        name
        for name in required
        if name not in globals()
    ]

    return {
        "module": "services.documents",
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
    print("DOCUMENTS.PY V3 - SELF TEST")
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

    print("=" * 60)
