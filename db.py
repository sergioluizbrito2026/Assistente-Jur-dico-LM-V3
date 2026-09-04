"""
Assistente Jurídico SaaS IA V3
db.py

Camada central de persistência SQLite.

Responsabilidades:
- conexão SQLite;
- inicialização do banco;
- criação das tabelas;
- índices de performance;
- seed demonstrativo;
- integridade referencial;
- compatibilidade com os serviços V3.

Arquitetura:

    Streamlit
        ↓
       db.py
        ↓
     SQLite
        ↓
    Serviços V3
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import sqlite3


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = (
    BASE_DIR
    / "database"
    / "app.db"
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONEXÃO SQLITE
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Abre uma conexão SQLite otimizada.

    Configurações:
    - WAL;
    - foreign keys;
    - busy timeout;
    - row factory.
    """

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    # Integridade referencial
    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    # Melhor concorrência
    conn.execute(
        "PRAGMA journal_mode = WAL"
    )

    conn.execute(
        "PRAGMA synchronous = NORMAL"
    )

    # Evita erro imediato em concorrência
    conn.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return conn


# ============================================================
# INICIALIZAÇÃO
# ============================================================

def init_db() -> None:
    """
    Cria todas as tabelas e índices necessários.

    Pode ser executado várias vezes sem destruir dados.
    """

    with get_connection() as c:

        c.executescript(
            """
            PRAGMA foreign_keys = ON;

            -- =================================================
            -- ORGANIZAÇÕES
            -- =================================================

            CREATE TABLE IF NOT EXISTS organizations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                plan TEXT DEFAULT 'Profissional',
                created_at TEXT NOT NULL
            );

            -- =================================================
            -- USUÁRIOS
            -- =================================================

            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY(organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE
            );

            -- =================================================
            -- DOCUMENTOS
            -- =================================================

            CREATE TABLE IF NOT EXISTS documents(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT,
                status TEXT,
                pages INTEGER DEFAULT 0,
                chunks INTEGER DEFAULT 0,
                ocr_pages INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,

                FOREIGN KEY(organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE
            );

            -- =================================================
            -- CHUNKS
            -- =================================================

            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                organization_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER,
                token_estimate INTEGER,
                metadata TEXT,

                FOREIGN KEY(document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE,

                FOREIGN KEY(organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE
            );

            -- =================================================
            -- PROCESSOS / CASOS
            -- =================================================

            CREATE TABLE IF NOT EXISTS cases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                client TEXT,
                category TEXT,
                priority TEXT,
                status TEXT DEFAULT 'Em andamento',
                created_at TEXT NOT NULL,

                FOREIGN KEY(organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE
            );

            -- =================================================
            -- AUDITORIA
            -- =================================================

            CREATE TABLE IF NOT EXISTS audit_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER,
                user_id INTEGER,
                action TEXT,
                entity_type TEXT,
                entity_id INTEGER,
                metadata TEXT,
                created_at TEXT NOT NULL,

                FOREIGN KEY(organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE,

                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE SET NULL
            );

            -- =================================================
            -- ÍNDICES
            -- =================================================

            CREATE INDEX IF NOT EXISTS idx_users_org
                ON users(organization_id);

            CREATE INDEX IF NOT EXISTS idx_users_email
                ON users(email);

            CREATE INDEX IF NOT EXISTS idx_documents_org
                ON documents(organization_id);

            CREATE INDEX IF NOT EXISTS idx_documents_org_name
                ON documents(
                    organization_id,
                    name
                );

            CREATE INDEX IF NOT EXISTS idx_chunks_org
                ON chunks(organization_id);

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);

            CREATE INDEX IF NOT EXISTS idx_chunks_org_document
                ON chunks(
                    organization_id,
                    document_id
                );

            CREATE INDEX IF NOT EXISTS idx_cases_org
                ON cases(organization_id);

            CREATE INDEX IF NOT EXISTS idx_cases_org_status
                ON cases(
                    organization_id,
                    status
                );

            CREATE INDEX IF NOT EXISTS idx_audit_org
                ON audit_logs(organization_id);

            CREATE INDEX IF NOT EXISTS idx_audit_user
                ON audit_logs(user_id);

            CREATE INDEX IF NOT EXISTS idx_audit_created
                ON audit_logs(created_at);
            """
        )


# ============================================================
# ORGANIZAÇÃO
# ============================================================

def get_organization(
    org_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Retorna uma organização pelo ID.
    """

    try:
        org_id = int(org_id)
    except (
        TypeError,
        ValueError,
    ):
        return None

    with get_connection() as c:

        row = c.execute(
            """
            SELECT *
            FROM organizations
            WHERE id = ?
            LIMIT 1
            """,
            (org_id,),
        ).fetchone()

    return dict(row) if row else None


# ============================================================
# USUÁRIO
# ============================================================

def get_user_by_email(
    email: str,
) -> Optional[Dict[str, Any]]:
    """
    Busca usuário pelo e-mail.
    """

    email = str(
        email or ""
    ).strip().lower()

    if not email:
        return None

    with get_connection() as c:

        row = c.execute(
            """
            SELECT
                id,
                organization_id,
                name,
                email,
                password_hash,
                role,
                created_at
            FROM users
            WHERE LOWER(email) = ?
            LIMIT 1
            """,
            (email,),
        ).fetchone()

    return dict(row) if row else None


# ============================================================
# SEED DEMO
# ============================================================

def seed_demo() -> None:
    """
    Cria dados demonstrativos apenas quando ainda não existem.

    O seed é idempotente:
    executar várias vezes não deve duplicar:
    - usuários;
    - documentos;
    - casos.

    O índice FAISS somente é reconstruído quando documentos
    demonstrativos realmente são inseridos.
    """

    from security.passwords import hash_password
    from services.embeddings import build_index_for_org

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    needs_index = False

    # ========================================================
    # ORGANIZAÇÃO + DADOS
    # ========================================================

    with get_connection() as c:

        # ----------------------------------------------------
        # Organização demo
        # ----------------------------------------------------

        c.execute(
            """
            INSERT OR IGNORE INTO organizations(
                id,
                name,
                plan,
                created_at
            )
            VALUES(1, ?, ?, ?)
            """,
            (
                "Alpha Advogados",
                "Profissional",
                now,
            ),
        )

        org_row = c.execute(
            """
            SELECT id
            FROM organizations
            WHERE id = 1
            LIMIT 1
            """
        ).fetchone()

        if not org_row:
            return

        org = int(
            org_row["id"]
        )

        # ----------------------------------------------------
        # Usuário administrador demo
        # ----------------------------------------------------

        user_exists = c.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            LIMIT 1
            """,
            (
                "admin@demo.local",
            ),
        ).fetchone()

        if not user_exists:

            c.execute(
                """
                INSERT INTO users(
                    organization_id,
                    name,
                    email,
                    password_hash,
                    role,
                    created_at
                )
                VALUES(?,?,?,?,?,?)
                """,
                (
                    org,
                    "Dr. João Silva",
                    "admin@demo.local",
                    hash_password(
                        "admin123"
                    ),
                    "Administrador",
                    now,
                ),
            )

        # ----------------------------------------------------
        # Documentos demonstrativos
        # ----------------------------------------------------

        demos = [
            (
                "Contrato_Prestacao_Servicos.pdf",
                "PDF",
                3,
                True,
                [
                    (
                        1,
                        (
                            "A cláusula prevê rescisão unilateral "
                            "e estabelece condições para "
                            "encerramento do contrato. Recomenda-se "
                            "verificar aviso prévio e penalidades "
                            "aplicáveis."
                        ),
                        0,
                    ),
                    (
                        2,
                        (
                            "O contrato possui cláusula de "
                            "confidencialidade e proteção de dados. "
                            "Devem ser validados escopo, "
                            "responsabilidades e medidas de segurança."
                        ),
                        1,
                    ),
                    (
                        3,
                        (
                            "A vigência contratual deve ser "
                            "interpretada em conjunto com as regras "
                            "de renovação e rescisão previstas "
                            "no instrumento."
                        ),
                        2,
                    ),
                ],
            ),
            (
                "Peticao_Inicial.txt",
                "TXT",
                1,
                False,
                [
                    (
                        1,
                        (
                            "A parte autora formula pedido de "
                            "cobrança com fundamento em obrigação "
                            "contratual e apresenta documentos "
                            "como elementos de suporte."
                        ),
                        0,
                    )
                ],
            ),
        ]

        for (
            name,
            typ,
            pages,
            ocr,
            demo_chunks,
        ) in demos:

            existing_doc = c.execute(
                """
                SELECT id
                FROM documents
                WHERE organization_id = ?
                  AND name = ?
                LIMIT 1
                """,
                (
                    org,
                    name,
                ),
            ).fetchone()

            if existing_doc:
                continue

            # ------------------------------------------------
            # Documento
            # ------------------------------------------------

            cur = c.execute(
                """
                INSERT INTO documents(
                    organization_id,
                    name,
                    type,
                    status,
                    pages,
                    chunks,
                    ocr_pages,
                    created_at
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    org,
                    name,
                    typ,
                    "Indexado",
                    pages,
                    len(demo_chunks),
                    1 if ocr else 0,
                    now,
                ),
            )

            document_id = cur.lastrowid

            # ------------------------------------------------
            # Chunks
            # ------------------------------------------------

            for (
                page,
                content,
                chunk_index,
            ) in demo_chunks:

                c.execute(
                    """
                    INSERT INTO chunks(
                        document_id,
                        organization_id,
                        content,
                        page,
                        chunk_index,
                        token_estimate,
                        metadata
                    )
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        document_id,
                        org,
                        content,
                        page,
                        chunk_index,
                        max(
                            1,
                            len(content) // 4,
                        ),
                        "{}",
                    ),
                )

            needs_index = True

        # ----------------------------------------------------
        # Caso demonstrativo
        # ----------------------------------------------------

        case_exists = c.execute(
            """
            SELECT id
            FROM cases
            WHERE organization_id = ?
              AND title = ?
            LIMIT 1
            """,
            (
                org,
                "Ação de Cobrança",
            ),
        ).fetchone()

        if not case_exists:

            c.execute(
                """
                INSERT INTO cases(
                    organization_id,
                    title,
                    client,
                    category,
                    priority,
                    status,
                    created_at
                )
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    org,
                    "Ação de Cobrança",
                    "Cliente Exemplo",
                    "Cível",
                    "Alta",
                    "Em andamento",
                    now,
                ),
            )

    # ========================================================
    # INDEXAÇÃO
    # ========================================================

    if needs_index:

        try:

            build_index_for_org(
                org
            )

        except Exception:
            # O banco não deve deixar de funcionar
            # caso embeddings/FAISS estejam indisponíveis.
            pass


# ============================================================
# HEALTH CHECK
# ============================================================

def database_health() -> Dict[str, Any]:
    """
    Verifica se o banco está acessível.
    """

    try:

        with get_connection() as c:

            row = c.execute(
                "SELECT 1 AS ok"
            ).fetchone()

            organizations = c.execute(
                "SELECT COUNT(*) FROM organizations"
            ).fetchone()[0]

            users = c.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0]

            documents = c.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

            chunks = c.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

            cases = c.execute(
                "SELECT COUNT(*) FROM cases"
            ).fetchone()[0]

        return {
            "status": "ok",
            "database": str(DB_PATH),
            "connection": bool(
                row and row["ok"] == 1
            ),
            "organizations": organizations,
            "users": users,
            "documents": documents,
            "chunks": chunks,
            "cases": cases,
        }

    except Exception as exc:

        return {
            "status": "error",
            "database": str(DB_PATH),
            "connection": False,
            "error": (
                f"{type(exc).__name__}: "
                f"{str(exc)[:300]}"
            ),
        }


# ============================================================
# TESTE DO MÓDULO
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural do banco.
    """

    required = [
        "get_connection",
        "init_db",
        "seed_demo",
        "get_organization",
        "get_user_by_email",
        "database_health",
    ]

    missing = [
        name
        for name in required
        if name not in globals()
    ]

    return {
        "module": "db.py",
        "status": (
            "ok"
            if not missing
            else "error"
        ),
        "database": str(DB_PATH),
        "required_functions": required,
        "missing_functions": missing,
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    init_db()

    result = self_test()

    print("=" * 60)
    print("DB.PY V3 - SELF TEST")
    print("=" * 60)

    print(
        f"Status: {result['status']}"
    )

    print(
        f"Banco: {result['database']}"
    )

    print(
        f"Funções obrigatórias: "
        f"{len(result['required_functions'])}"
    )

    print(
        f"Funções ausentes: "
        f"{result['missing_functions']}"
    )

    health = database_health()

    print(
        f"Database health: "
        f"{health['status']}"
    )

