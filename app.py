"""
Assistente Jurídico SaaS IA V3.1
Interface profissional — estilo SaaS jurídico

Pipeline:
RAG -> Retriever -> Reranker -> Guard -> Orchestrator
     -> LLM -> Citações -> Evaluation
"""


from __future__ import annotations

import inspect
import traceback
from typing import Any, Dict

import streamlit as st

from db import init_db, seed_demo

from services.audit import audit
from services.cases import create_case, list_cases
from services.documents import list_documents
from services.evaluation import evaluate_answer
from services.rag_pipeline import rag_answer, retrieve_and_rerank
from services.ai_orchestrator import orchestrate, risk_analysis
from services.auth import authenticate, get_current_user, logout


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Assistente Jurídico IA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BANCO DE DADOS
# ============================================================

from db import init_db, seed_demo


# ============================================================
# SERVIÇOS
# ============================================================

from services.audit import audit

from services.auth import (
    authenticate,
    get_current_user,
    logout,
)

from services.cases import (
    create_case,
    list_cases,
)

from services.documents import list_documents

from services.evaluation import evaluate_answer

from services.ingestion import ingest_document

from services.rag_pipeline import (
    rag_answer,
    retrieve_and_rerank,
)

from services.ai_orchestrator import (
    orchestrate,
    risk_analysis,
)


# ============================================================
# BANCO DE DADOS — INICIALIZAÇÃO
# ============================================================

try:
    init_db()

except Exception as exc:
    st.error("Erro ao inicializar o banco de dados.")

    with st.expander("Detalhes técnicos"):
        st.code(
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
        )

    st.stop()


# ============================================================
# DADOS DEMONSTRATIVOS
# ============================================================

try:
    seed_demo()

except Exception as exc:
    st.warning(
        "O banco foi inicializado, mas os dados "
        "demonstrativos não puderam ser carregados."
    )

    with st.expander("Detalhes técnicos"):
        st.code(
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
        )




# ============================================================

st.markdown(
    """
<style>

:root {
    --navy:#07152b;
    --navy2:#0b1d38;
    --blue:#1769e0;
    --blue2:#0f5bd7;
    --gold:#d7a94b;
    --bg:#f5f7fb;
    --card:#ffffff;
    --text:#142033;
    --muted:#6c7890;
    --border:#e4e9f2;
    --green:#19a463;
    --orange:#f59e0b;
    --red:#dc3545;
}

.stApp {
    background:var(--bg);
    color:var(--text);
}

[data-testid="stHeader"] {
    background:transparent;
}

.block-container {
    max-width:1480px;
    padding-top:1.2rem;
    padding-bottom:3rem;
}

[data-testid="stSidebar"] {
    background:linear-gradient(
        180deg,
        #07152b 0%,
        #091b35 100%
    );

    border-right:1px solid #183253;
}

[data-testid="stSidebar"] * {
    color:#eef4ff;
}

[data-testid="stSidebar"] .stButton > button {
    background:transparent;
    border:0;
    color:#eef4ff;
    text-align:left;
    border-radius:10px;
    padding:0.62rem 0.8rem;
    font-size:0.94rem;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background:#12335e;
    color:white;
}

.sidebar-brand {
    padding:0.25rem 0.3rem 1rem;
    border-bottom:1px solid #193454;
    margin-bottom:1rem;
}

.sidebar-brand .logo {
    font-size:2rem;
    color:#f2c66d;
}

.sidebar-brand .title {
    font-size:1.12rem;
    font-weight:700;
}

.sidebar-brand .sub {
    font-size:0.72rem;
    color:#a9bad3;
}

.sidebar-plan {
    margin-top:1rem;
    padding:1rem;
    border:1px solid #23456e;
    border-radius:14px;
    background:linear-gradient(
        145deg,
        #102b4d,
        #0b203d
    );
}

.sidebar-plan .value {
    font-size:1.2rem;
    font-weight:700;
}

.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:0.25rem 0 1rem;
}

.brand-title {
    font-size:1.5rem;
    font-weight:750;
    letter-spacing:-0.02em;
}

.version-pill {
    display:inline-block;
    margin-left:0.55rem;
    padding:0.28rem 0.62rem;
    border-radius:999px;
    background:#eaf1ff;
    color:#245bc3;
    font-size:0.72rem;
    font-weight:700;
}

.hero {
    background:white;
    border:1px solid var(--border);
    border-radius:18px;
    padding:1.45rem;
    box-shadow:0 6px 24px rgba(20,32,51,.05);
}

.hero h1 {
    margin:0;
    font-size:2rem;
    letter-spacing:-0.035em;
}

.hero p {
    color:var(--muted);
    margin-top:0.35rem;
}

.card {
    background:white;
    border:1px solid var(--border);
    border-radius:16px;
    padding:1.1rem;
    box-shadow:0 5px 20px rgba(20,32,51,.045);
}

.metric-card {
    background:white;
    border:1px solid var(--border);
    border-radius:15px;
    padding:1rem;
    min-height:110px;
    box-shadow:0 4px 16px rgba(20,32,51,.04);
}

.metric-label {
    color:var(--muted);
    font-size:0.78rem;
    font-weight:600;
}

.metric-value {
    font-size:1.65rem;
    font-weight:750;
    margin-top:0.35rem;
}

.metric-trend {
    color:var(--green);
    font-size:0.72rem;
    margin-top:0.15rem;
}

.section-title {
    font-size:1.08rem;
    font-weight:750;
    margin:0.2rem 0 0.8rem;
}

.question-chip {
    display:inline-block;
    border:1px solid var(--border);
    border-radius:999px;
    padding:0.42rem 0.72rem;
    margin:0.15rem;
    background:#fff;
    color:#46536b;
    font-size:0.76rem;
}

.status-ok {
    color:var(--green);
    font-weight:700;
}

.risk-high {
    color:var(--red);
    font-weight:750;
}

.risk-medium {
    color:#d97706;
    font-weight:750;
}

.risk-low {
    color:var(--green);
    font-weight:750;
}

.chat-panel {
    background:white;
    border:1px solid var(--border);
    border-radius:18px;
    padding:1rem;
    box-shadow:0 5px 20px rgba(20,32,51,.04);
}

.citation {
    background:#f7f9fd;
    border:1px solid var(--border);
    border-radius:10px;
    padding:0.55rem 0.7rem;
    margin:0.35rem 0;
}

.risk-box {
    border-radius:14px;
    padding:1rem;
    background:#fff7f7;
    border:1px solid #ffd7db;
}

.footer-note {
    color:#8792a6;
    font-size:0.72rem;
    text-align:center;
    margin-top:2rem;
}

div[data-testid="stMetric"] {
    background:white;
    border:1px solid var(--border);
    border-radius:15px;
    padding:0.75rem;
}

div[data-testid="stFileUploader"] {
    background:white;
    border:1px dashed #bdc8d9;
    border-radius:14px;
    padding:0.4rem;
}

.stButton > button[kind="primary"] {
    border-radius:10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_dict(value: Any) -> Dict[str, Any]:
    """
    Garante que o resultado de um serviço seja convertido
    para um dicionário seguro.
    """

    if isinstance(value, dict):
        return value

    return {
        "answer": str(value or ""),
        "citations": [],
        "retrieved": [],
        "reranked": [],
    }


def safe_list(value: Any) -> list:
    """
    Garante uma lista segura.
    """

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return []


def call_orchestrator(
    query: str,
    org_id: Any,
    mode: str = "auto",
    top_k: int = 8,
    rerank_k: int = 5,
    extra_context: str | None = None,
) -> Dict[str, Any]:
    """
    Chama o orchestrator de forma compatível com diferentes
    assinaturas da função orchestrate().
    """

    query = str(query or "").strip()

    if not query:
        return {
            "answer": "Informe uma pergunta.",
            "citations": [],
            "retrieved": [],
            "reranked": [],
            "agent": "none",
            "intent": "empty",
        }

    kwargs = {
        "query": query,
        "question": query,
        "org_id": org_id,
        "organization_id": org_id,
        "mode": mode,
        "top_k": top_k,
        "rerank_k": rerank_k,
        "extra_context": extra_context,
    }

    try:

        signature = inspect.signature(orchestrate)
        parameters = signature.parameters

        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters.values()
        )

        if accepts_kwargs:
            filtered = {
                k: v
                for k, v in kwargs.items()
                if v is not None
            }

        else:
            filtered = {
                k: v
                for k, v in kwargs.items()
                if k in parameters and v is not None
            }

        # Evita enviar query e question simultaneamente
        if "query" in parameters:
            filtered.pop("question", None)

        elif "question" in parameters:
            filtered.pop("query", None)

        # Evita enviar org_id e organization_id simultaneamente
        if "org_id" in parameters:
            filtered.pop("organization_id", None)

        elif "organization_id" in parameters:
            filtered.pop("org_id", None)

        result = safe_dict(
            orchestrate(**filtered)
        )

        result.setdefault("answer", "")
        result.setdefault("citations", [])
        result.setdefault("retrieved", [])
        result.setdefault("reranked", [])
        result.setdefault("agent", "juridico")
        result.setdefault("intent", "legal_query")

        return result

    except Exception as exc:

        return {
            "answer": "",
            "citations": [],
            "retrieved": [],
            "reranked": [],
            "agent": "error",
            "intent": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def render_citations(citations):
    """
    Renderiza as evidências recuperadas.
    """

    citations = safe_list(citations)

    if not citations:
        return

    st.markdown("#### 📚 Evidências e citações")

    for i, citation in enumerate(citations, 1):

        if not isinstance(citation, dict):
            continue

        cid = citation.get("id", i)

        doc = citation.get(
            "document",
            citation.get(
                "document_name",
                "Documento",
            ),
        )

        page = citation.get(
            "page",
            "N/D",
        )

        chunk = citation.get(
            "chunk_id",
            "N/D",
        )

        content = citation.get(
            "content",
            citation.get(
                "text",
                "",
            ),
        )

        st.markdown(
            f"""
            <div class="citation">
                <b>[{cid}] {doc}</b>
                · página {page}
                <br>
                <small>chunk: {chunk}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if content:

            with st.expander(
                "Ver trecho da evidência"
            ):
                st.write(content)


def render_diagnostic(result):
    """
    Mostra diagnóstico técnico da execução.
    Versão segura contra SyntaxError em f-strings.
    """

    result = safe_dict(result)

    with st.expander(
        "🔍 Diagnóstico da execução",
        expanded=False,
    ):

        c1, c2, c3 = st.columns(3)

        agent = str(
            result.get("agent", "N/D")
        )

        intent = str(
            result.get("intent", "N/D")
        )

        citations = safe_list(
            result.get("citations")
        )

        evidence_count = result.get(
            "evidence_count",
            len(citations),
        )

        retrieved = safe_list(
            result.get("retrieved")
        )

        reranked = safe_list(
            result.get("reranked")
        )

        latency = result.get(
            "latency_ms",
            "N/D",
        )

        c1.write(
            "**Agente:** " + agent
        )

        c2.write(
            "**Intent:** " + intent
        )

        c3.write(
            "**Evidências:** "
            + str(evidence_count)
        )

        c1.write(
            "**Documentos recuperados:** "
            + str(len(retrieved))
        )

        c2.write(
            "**Chunks reranked:** "
            + str(len(reranked))
        )

        c3.write(
            "**Latência:** "
            + str(latency)
            + " ms"
        )

        guard = result.get(
            "guard",
            {},
        )

        if isinstance(guard, dict):

            allowed = guard.get(
                "allowed",
                True,
            )

            if allowed:
                st.write(
                    "**Guard Agent:** 🟢 Permitido"
                )
            else:
                st.write(
                    "**Guard Agent:** 🔴 Bloqueado"
                )

        reason = result.get("reason")

        if reason:
            st.info(
                str(reason)
            )

        error = result.get("error")

        if error:
            st.error(
                str(error)
            )



# ============================================================
# LOGIN
# ============================================================

user = get_current_user()

if not user:

    st.markdown(
        """
        <div style="
            max-width:520px;
            margin:7rem auto 0;
            text-align:center;
        ">
            <div style="font-size:3.2rem;">
                ⚖️
            </div>

            <h1>
               st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1>Assistente Jurídico IA</h1>
        <p style="color:#6c7890;">
            Inteligência artificial para documentos,
            riscos, pesquisas e análises jurídicas.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

    with st.form("login"):

        email = st.text_input(
            "E-mail",
            "admin@demo.local",
        )

        password = st.text_input(
            "Senha",
            "admin123",
            type="password",
        )

        submitted = st.form_submit_button(
            "Entrar",
            type="primary",
            use_container_width=True,
        )

        if submitted:

            try:

                if authenticate(
                    email,
                    password,
                ):
                    st.rerun()

                else:
                    st.error(
                        "Credenciais inválidas."
                    )

            except Exception as exc:

                st.error(
                    f"Erro durante autenticação: {exc}"
                )

    st.info(
        "Demo: admin@demo.local / admin123"
    )

    st.stop()


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []


File "/mount/src/assistente-jur-dico-lm-v3/app.py", line 808
              <div class="logo">⚖️</div>
                                ^
SyntaxError: invalid character '⚖' (U+2696)



# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("Olá, Dr. 👋")
    st.markdown("### Como posso ajudar você hoje?")

    st.markdown("<br>", unsafe_allow_html=True)

   # ============================================================
    # MÉTRICAS REAIS
    # ============================================================

    try:
        from db import get_connection

        with get_connection() as c:
            org_id = user.get("organization_id")

            documents_count = c.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE organization_id=?
                """,
                (org_id,),
            ).fetchone()[0]

            cases_count = c.execute(
                """
                SELECT COUNT(*)
                FROM cases
                WHERE organization_id=?
                """,
                (org_id,),
            ).fetchone()[0]

            chunks_count = c.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE organization_id=?
                """,
                (org_id,),
            ).fetchone()[0]

            audit_count = c.execute(
                """
                SELECT COUNT(*)
                FROM audit_logs
                WHERE organization_id=?
                """,
                (org_id,),
            ).fetchone()[0]

    except Exception:
        documents_count = 0
        cases_count = 0
        chunks_count = 0
        audit_count = 0

    metrics = [
        ("▣", "Processos", str(cases_count), "registros"),
        ("□", "Documentos", str(documents_count), "indexados"),
        ("⌕", "Chunks", str(chunks_count), "na base vetorial"),
        ("◈", "Eventos", str(audit_count), "auditoria"),
    ]

    cols = st.columns(4)

    for col, metric in zip(cols, metrics):
        icon, label, value, trend = metric

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.25rem;">
                        {icon}
                    </div>
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                    <div class="metric-trend">
                        {trend}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2.1, 1])

    # --------------------------------------------------------
    # CENTRAL
    # --------------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="card">

                <div class="section-title">
                    Pergunte algo sobre seus
                    documentos ou sobre direito...
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        dash_q = st.text_input(
            "Pergunta jurídica",
            placeholder=(
                "Ex.: Quem é a CONTRATANTE?"
            ),
            label_visibility="collapsed",
            key="dashboard_question",
        )

        a, b, c = st.columns(
            [1, 1, 1.3]
        )

        with a:

            attach = st.button(
                "📎 Anexar documentos",
                use_container_width=True,
            )

        with b:

            research = st.button(
                "⚖️ Pesquisa Jurídica",
                use_container_width=True,
            )

        with c:

            risk = st.button(
                "🛡️ Análise de Risco",
                use_container_width=True,
            )

        if dash_q:

            if st.button(
                "➤ Perguntar",
                type="primary",
            ):

                st.session_state.page = (
                    "Assistente IA"
                )

                st.session_state.pending_question = (
                    dash_q
                )

                st.rerun()

        if attach:

            st.session_state.page = (
                "Documentos"
            )

            st.rerun()

        if research:

            st.session_state.page = (
                "Pesquisas Jurídicas"
            )

            st.rerun()

        if risk:

            st.session_state.page = (
                "Análise de Risco"
            )

            st.rerun()

        st.markdown(
            """
            <div style="margin-top:.7rem;">

                <span class="question-chip">
                    Quais são os riscos deste contrato?
                </span>

                <span class="question-chip">
                    Resuma a petição inicial
                </span>

                <span class="question-chip">
                    Qual o prazo para recurso?
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

     st.markdown(
        """
        <div class="section-title">
            &#128196; Documentos Recentes
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        documents = list_documents(
            user.get("organization_id")
        )

        if not documents:
            st.info(
                "Nenhum documento encontrado."
            )

        for document in documents[:5]:
            if not isinstance(
                document,
                dict,
            ):
                continue

            with st.container(
                border=True
            ):
                c1, c2, c3 = st.columns(
                    [0.1, 0.72, 0.18]
                )

                with c1:
                    st.markdown("&#128196;", unsafe_allow_html=True)

                with c2:
                    st.write(
                        f"**{document.get('name', 'Documento')}**"
                    )

                    st.caption(
                        f"{document.get('type', 'N/D')} · "
                        f"{document.get('pages', 0)} páginas · "
                        f"{document.get('chunks', 0)} chunks"
                    )

                with c3:
                    status = document.get(
                        "status",
                        "N/D",
                    )

                    if status == "Indexado":
                        st.success(status)
                    else:
                        st.warning(status)

    except Exception as exc:
        st.error(
            f"Erro ao carregar documentos: {exc}"
        )

  # ============================================================
# LATERAL DASHBOARD
# ============================================================

with right if 'right' in locals() else st.sidebar:

    st.markdown(
        """
        <div class="card">
            <div class="section-title">
                Analise de Risco IA
            </div>
            <div style="font-size:3rem; font-weight:800; text-align:center; margin:1rem 0;">
                IA
            </div>
            <div style="text-align:center; color:#6c7890;">
                analise disponivel
            </div>
            <hr>
            <div>
                Alto risco
                <b style="float:right;">-</b>
            </div>
            <div>
                Medio risco
                <b style="float:right;">-</b>
            </div>
            <div>
                Baixo risco
                <b style="float:right;">-</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="section-title">
                Pesquisas Juridicas Recentes
            </div>
            <p>
                <b>Jurisprudencia sobre danos morais</b>
                <br>
                <small>Fonte a consultar</small>
            </p>
            <hr>
            <p>
                <b>Prazo para interposicao de recurso</b>
                <br>
                <small>Fonte a consultar</small>
            </p>
            <hr>
            <p>
                <b>Responsabilidade civil do fornecedor</b>
                <br>
                <small>Fonte a consultar</small>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="section-title">
                Status da Plataforma
            </div>
            <p>Banco de dados operacional</p>
            <p>RAG disponivel</p>
            <p>Retriever/Reranker disponivel</p>
            <p>Orchestrator carregado</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
            Assistente Juridico IA &middot;
            RAG + Multiagentes + Evidencias &middot; V3.1
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# ASSISTENTE IA
# ============================================================

elif page == "Assistente IA" if 'page' in locals() else True:

    st.markdown(
        """
        <div class="hero">
            <h1>Assistente Juridico IA</h1>
            <p>
                Analise documentos, consulte sua base juridica
                e obtenha respostas fundamentadas em evidencias.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Historico
    if 'messages' in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.session_state.messages = []

    pending = st.session_state.pop(
        "pending_question",
        None,
    )

    q = st.chat_input(
        "Pergunte sobre seus documentos ou sobre direito..."
    )

    q = q or pending

    if q:
        q = str(q).strip()

        if q:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )

            with st.chat_message("user"):
                st.markdown(q)

            with st.chat_message("assistant"):
                with st.spinner(
                    "Analisando -> Guard -> RAG -> "
                    "Retriever -> Reranker -> Agente -> LLM..."
                ):
                    try:
                        result = call_orchestrator(
                            query=q,
                            org_id=user.get("organization_id") if 'user' in locals() else None,
                            mode="auto",
                            top_k=8,
                            rerank_k=5,
                        )
                    except Exception as e:
                        result = {"answer": f"Erro ao executar o orquestrador: {e}"}

                response = str(
                    result.get("answer", "") or ""
                ).strip()

                if response:
                    st.markdown(response)
                else:
                    st.warning(
                        "A execucao terminou sem uma "
                        "resposta textual. As evidencias "
                        "recuperadas permanecem disponiveis abaixo."
                    )

                if 'render_citations' in globals():
                    render_citations(result.get("citations"))

                if 'render_diagnostic' in globals():
                    render_diagnostic(result)

                if result.get("error"):
                    with st.expander("Detalhes tecnicos"):
                        st.code(str(result["error"]))

                # Auditoria
                try:
                    if 'audit' in globals():
                        audit(
                            user if 'user' in locals() else {},
                            "rag.ask",
                            "conversation",
                            None,
                            {
                                "query": q,
                                "agent": result.get("agent"),
                                "intent": result.get("intent"),
                            },
                        )
                except Exception:
                    pass

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        response
                        or "Execucao sem resposta textual."
                    ),
                }
            )


# ============================================================
# DOCUMENTOS
# ============================================================

elif page == "Documentos":

    st.markdown(
        """
        <div class="hero">
            <h1>📄 Documentos</h1>
            <p>
                Centralize contratos, petições,
                procurações e demais documentos jurídicos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    up = st.file_uploader(
        "Adicionar documento",
        type=["pdf", "docx", "txt"],
    )

    use_ocr = st.checkbox(
        "Usar OCR quando necessário",
        True,
    )

    if up:
        if st.button("🚀 Processar e indexar", type="primary"):
            with st.status("Processando documento...", expanded=True) as status:
                try:
                    st.write("Extraindo texto e páginas...")
                    st.write("Aplicando OCR quando necessário...")
                    st.write("Gerando chunks...")
                    st.write("Gerando embeddings...")
                    st.write("Atualizando índice vetorial...")

                    result = safe_dict(
                        ingest_document(
                            up,
                            user.get("organization_id") if 'user' in locals() else None,
                            use_ocr=use_ocr,
                        )
                    ) if 'ingest_document' in globals() else {}

                    status.update(
                        label="Documento processado com sucesso",
                        state="complete",
                    )

                    st.success(
                        f"{result.get('chunks', 0)} chunks - "
                        f"{result.get('pages', 0)} páginas - "
                        f"OCR: {result.get('ocr_pages', 0)} páginas"
                    )

                except Exception as exc:
                    status.update(
                        label="Falha no processamento",
                        state="error",
                    )
                    st.error(f"Erro: {exc}")
                    st.code(traceback.format_exc())

    st.markdown("### Documentos disponíveis")

    try:
        documents = list_documents(user.get("organization_id") if 'user' in locals() else None) if 'list_documents' in globals() else []

        if not documents:
            st.info("Nenhum documento encontrado.")

        for document in documents:
            if not isinstance(document, dict):
                continue

            with st.container(border=True):
                c1, c2, c3 = st.columns([0.08, 0.72, 0.2])

                with c1:
                    st.markdown("📄")

                with c2:
                    st.write(f"**{document.get('name', 'Documento')}**")
                    st.caption(
                        f"{document.get('type', 'N/D')} - "
                        f"{document.get('status', 'N/D')} - "
                        f"{document.get('pages', 0)} páginas - "
                        f"{document.get('chunks', 0)} chunks"
                    )

                with c3:
                    st.write(document.get("status", "N/D"))

    except Exception as exc:
        st.error(f"Erro ao carregar documentos: {exc}")


# ============================================================
# ANÁLISE DE RISCO
# ============================================================

elif page == "Análise de Risco":

    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ Análise de Risco IA</h1>
            <p>Agente de Risco + RAG + evidências documentais.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    text = st.text_area(
        "Cole o texto ou resumo do documento",
        height=260,
        placeholder="Cole aqui o conteúdo do contrato, petição ou documento...",
    )

    if st.button("🔍 Analisar riscos", type="primary"):
        if not text.strip():
            st.warning("Informe o texto para análise.")
        else:
            with st.spinner("Agente de Risco analisando evidências..."):
                try:
                    result = safe_dict(
                        risk_analysis(
                            query="Analise os principais riscos jurídicos...",
                            org_id=user.get("organization_id") if 'user' in locals() else None,
                            extra_context=text,
                        )
                    ) if 'risk_analysis' in globals() else {}

                    answer = str(result.get("answer", "") or "").strip()
                    if answer:
                        st.markdown(answer)
                    else:
                        st.warning("O agente terminou sem resposta textual.")

                    if 'render_citations' in globals():
                        render_citations(result.get("citations"))
                    if 'render_diagnostic' in globals():
                        render_diagnostic(result)

                except Exception as exc:
                    st.error(f"Erro na análise de risco: {exc}")
                    st.code(traceback.format_exc())


# ============================================================
# CONFIGURAÇÕES
# ============================================================

elif page == "Configurações":

    st.markdown(
        """
        <div class="hero">
            <h1>⚙️ Configurações</h1>
            <p>Controle LLM, RAG, segurança, agentes e parâmetros.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🤖 Inteligência Artificial")
        llm = st.selectbox("LLM", ["Modo Demo", "Gemini", "OpenAI"])
        top_k = st.number_input("Retriever Top-K", min_value=1, max_value=50, value=8)
        rerank_k = st.number_input("Reranker Top-K", min_value=1, max_value=20, value=5)

    with c2:
        st.markdown("### 🛡️ Segurança")
        citations_required = st.checkbox("Citações obrigatórias", True)
        evaluation_enabled = st.checkbox("Evaluation habilitado", True)
        guard_enabled = st.checkbox("Guard Agent habilitado", True)
