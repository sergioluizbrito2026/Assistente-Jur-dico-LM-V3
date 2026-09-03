from pathlib import Path
from datetime import datetime
import sqlite3


# ============================================================
# CONFIGURAÇÃO DO BANCO
# ============================================================

DB_PATH = Path(__file__).parent / "database" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONEXÃO OTIMIZADA SQLITE
# ============================================================

def get_connection():
    """
    Abre uma conexão SQLite otimizada para o SaaS.

    WAL permite leitura concorrente sem bloquear o banco durante
    pequenas operações de escrita.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    # Integridade referencial
    conn.execute("PRAGMA foreign_keys = ON")

    # Melhor concorrência entre Streamlit e operações de escrita
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    # Evita falha imediata quando o banco estiver temporariamente ocupado
    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


# ============================================================
# INICIALIZAÇÃO DO BANCO
# ============================================================

def init_db():
    """
    Cria as tabelas e índices necessários.
    É seguro executar em cada inicialização do Streamlit.
    """

    with get_connection() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS organizations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                plan TEXT DEFAULT 'Profissional',
                created_at TEXT NOT NULL
            );

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
            );

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
            );

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
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                client TEXT,
                category TEXT,
                priority TEXT,
                status TEXT DEFAULT 'Em andamento',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER,
                user_id INTEGER,
                action TEXT,
                entity_type TEXT,
                entity_id INTEGER,
                metadata TEXT,
                created_at TEXT NOT NULL
            );

            -- =================================================
            -- ÍNDICES DE PERFORMANCE
            -- =================================================

            CREATE INDEX IF NOT EXISTS idx_users_org
                ON users(organization_id);

            CREATE INDEX IF NOT EXISTS idx_documents_org
                ON documents(organization_id);

            CREATE INDEX IF NOT EXISTS idx_documents_org_name
                ON documents(organization_id, name);

            CREATE INDEX IF NOT EXISTS idx_chunks_org
                ON chunks(organization_id);

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);

            CREATE INDEX IF NOT EXISTS idx_chunks_org_document
                ON chunks(organization_id, document_id);

            CREATE INDEX IF NOT EXISTS idx_cases_org
                ON cases(organization_id);

            CREATE INDEX IF NOT EXISTS idx_audit_org
                ON audit_logs(organization_id);

            CREATE INDEX IF NOT EXISTS idx_audit_user
                ON audit_logs(user_id);

            CREATE INDEX IF NOT EXISTS idx_audit_created
                ON audit_logs(created_at);
            """
        )


# ============================================================
# SEED DEMO
# ============================================================

def seed_demo():
    """
    Cria somente os dados demonstrativos que ainda não existem.

    IMPORTANTE:
    O índice vetorial só é construído quando realmente existem
    chunks novos. Isso evita reconstruir o FAISS em todo rerun
    do Streamlit.
    """

    from security.passwords import hash_password
    from services.embeddings import build_index_for_org

    now = datetime.now().isoformat(timespec="seconds")
    needs_index = False

    with get_connection() as c:

        # ----------------------------------------------------
        # Organização demo
        # ----------------------------------------------------

        c.execute(
            """
            INSERT OR IGNORE INTO organizations(
                id, name, plan, created_at
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
            "SELECT id FROM organizations WHERE id = 1"
        ).fetchone()

        if not org_row:
            return

        org = org_row["id"]

        # ----------------------------------------------------
        # Usuário administrador demo
        # ----------------------------------------------------

        user_exists = c.execute(
            """
            SELECT 1
            FROM users
            WHERE email = ?
            LIMIT 1
            """,
            ("admin@demo.local",),
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
                    hash_password("admin123"),
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
                        "A cláusula prevê rescisão unilateral e estabelece "
                        "condições para encerramento do contrato. Recomenda-se "
                        "verificar aviso prévio e penalidades aplicáveis.",
                        0,
                    ),
                    (
                        2,
                        "O contrato possui cláusula de confidencialidade e "
                        "proteção de dados. Devem ser validados escopo, "
                        "responsabilidades e medidas de segurança.",
                        1,
                    ),
                    (
                        3,
                        "A vigência contratual deve ser interpretada em "
                        "conjunto com as regras de renovação e rescisão "
                        "previstas no instrumento.",
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
                        "A parte autora formula pedido de cobrança com "
                        "fundamento em obrigação contratual e apresenta "
                        "documentos como elementos de suporte.",
                        0,
                    )
                ],
            ),
        ]

        for name, typ, pages, ocr, demo_chunks in demos:

            existing_doc = c.execute(
                """
                SELECT id
                FROM documents
                WHERE organization_id = ?
                  AND name = ?
                LIMIT 1
                """,
                (org, name),
            ).fetchone()

            if existing_doc:
                continue

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

            did = cur.lastrowid

            for page, content, idx in demo_chunks:
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
                        did,
                        org,
                        content,
                        page,
                        idx,
                        max(1, len(content) // 4),
                        "{}",
                    ),
                )

            needs_index = True

        # ----------------------------------------------------
        # Processo demonstrativo
        # ----------------------------------------------------

        case_count = c.execute(
            """
            SELECT COUNT(*)
            FROM cases
            WHERE organization_id = ?
            """,
            (org,),
        ).fetchone()[0]

        if case_count == 0:
            c.execute(
                """
                INSERT INTO cases(
                    organization_id,
                    title,
                    client,
                    category,
                    priority,
                    created_at
                )
                VALUES(?,?,?,?,?,?)
                """,
                (
                    org,
                    "Ação de Cobrança",
                    "Cliente Exemplo",
                    "Cível",
                    "Alta",
                    now,
                ),
            )

    # --------------------------------------------------------
    # Só atualiza o índice quando houve documento novo
    # --------------------------------------------------------

    if needs_index:
        try:
            build_index_for_org(org)
        except Exception:
            # O banco continua funcionando mesmo se a camada
            # vetorial estiver indisponível.
            pass
