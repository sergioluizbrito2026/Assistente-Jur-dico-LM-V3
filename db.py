from pathlib import Path
from datetime import datetime
import sqlite3

DB_PATH = Path(__file__).parent / "database" / "app.db"
DB_PATH.parent.mkdir(exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    with get_connection() as c:
        c.executescript("""
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
            FOREIGN KEY(organization_id) REFERENCES organizations(id)
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
            FOREIGN KEY(organization_id) REFERENCES organizations(id)
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
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
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
        """)

def seed_demo():
    from security.passwords import hash_password
    from services.embeddings import build_index_for_org
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection() as c:
        if c.execute("SELECT COUNT(*) FROM organizations").fetchone()[0]:
            return
        c.execute("INSERT INTO organizations(name,plan,created_at) VALUES(?,?,?)", ("Alpha Advogados","Profissional",now))
        org = c.execute("SELECT id FROM organizations").fetchone()[0]
        c.execute(
            "INSERT INTO users(organization_id,name,email,password_hash,role,created_at) VALUES(?,?,?,?,?,?)",
            (org,"Dr. João Silva","admin@demo.local",hash_password("admin123"),"Administrador",now)
        )
        demos = [
            ("Contrato_Prestacao_Servicos.pdf","PDF",3,True,[
                (1,"A cláusula prevê rescisão unilateral e estabelece condições para encerramento do contrato. Recomenda-se verificar aviso prévio e penalidades aplicáveis.",0),
                (2,"O contrato possui cláusula de confidencialidade e proteção de dados. Devem ser validados escopo, responsabilidades e medidas de segurança.",1),
                (3,"A vigência contratual deve ser interpretada em conjunto com as regras de renovação e rescisão previstas no instrumento.",2),
            ]),
            ("Peticao_Inicial.txt","TXT",1,False,[
                (1,"A parte autora formula pedido de cobrança com fundamento em obrigação contratual e apresenta documentos como elementos de suporte.",0)
            ])
        ]
        for name,typ,pages,ocr,chunks in demos:
            c.execute(
                "INSERT INTO documents(organization_id,name,type,status,pages,chunks,ocr_pages,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (org,name,typ,"Indexado",pages,len(chunks),1 if ocr else 0,now)
            )
            did=c.lastrowid
            for page,content,idx in chunks:
                c.execute(
                    "INSERT INTO chunks(document_id,organization_id,content,page,chunk_index,token_estimate,metadata) VALUES(?,?,?,?,?,?,?)",
                    (did,org,content,page,idx,max(1,len(content)//4),"{}")
                )
        c.execute(
            "INSERT INTO cases(organization_id,title,client,category,priority,created_at) VALUES(?,?,?,?,?,?)",
            (org,"Ação de Cobrança","Cliente Exemplo","Cível","Alta",now)
        )
    try:
        build_index_for_org(org)
    except Exception:
        pass
