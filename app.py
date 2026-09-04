
"""
Assistente Jurídico SaaS IA V3.1
Interface profissional — estilo SaaS jurídico

Mantém o backend existente:
RAG -> Retriever -> Reranker -> Guard -> Orchestrator -> LLM -> Citações -> Evaluation
"""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Dict

import pandas as pd
import streamlit as st

from db import init_db, seed_demo
from services.audit import audit
from services.auth import authenticate, get_current_user, logout
from services.cases import create_case, list_cases
from services.documents import list_documents
from services.evaluation import evaluate_answer
from services.ingestion import ingest_document
from services.rag_pipeline import rag_answer, retrieve_and_rerank
from services.ai_orchestrator import orchestrate, risk_analysis


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Assistente Jurídico IA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    init_db()
    seed_demo()
except Exception as exc:
    st.error(f"Erro ao inicializar o banco: {exc}")
    st.stop()


# ============================================================
# DESIGN SYSTEM — inspirado no modelo enviado
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
    background:linear-gradient(180deg,#07152b 0%,#091b35 100%);
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
    background:linear-gradient(145deg,#102b4d,#0b203d);
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
    if isinstance(value, dict):
        return value
    return {
        "answer": str(value or ""),
        "citations": [],
        "retrieved": [],
        "reranked": [],
    }


def safe_list(value: Any) -> list:
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
            filtered = {k: v for k, v in kwargs.items() if v is not None}
        else:
            filtered = {
                k: v for k, v in kwargs.items()
                if k in parameters and v is not None
            }

        if "query" in parameters:
            filtered.pop("question", None)
        elif "question" in parameters:
            filtered.pop("query", None)

        if "org_id" in parameters:
            filtered.pop("organization_id", None)
        elif "organization_id" in parameters:
            filtered.pop("org_id", None)

        result = safe_dict(orchestrate(**filtered))

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
    citations = safe_list(citations)
    if not citations:
        return

    st.markdown("#### 📚 Evidências e citações")

    for i, c in enumerate(citations, 1):
        if not isinstance(c, dict):
            continue

        cid = c.get("id", i)
        doc = c.get("document", c.get("document_name", "Documento"))
        page = c.get("page", "N/D")
        chunk = c.get("chunk_id", "N/D")
        content = c.get("content", c.get("text", ""))

        st.markdown(
            f"""
            <div class="citation">
                <b>[{cid}] {doc}</b> · página {page}
                <br>
                <small>chunk: {chunk}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if content:
            with st.expander("Ver trecho da evidência"):
                st.write(content)


def render_diagnostic(result):
    with st.expander("🔍 Diagnóstico da execução", expanded=False):
        c1, c2, c3 = st.columns(3)

        c1.write(f"**Agente:** {result.get('agent', 'N/D')}")
        c2.write(f"**Intent:** {result.get('intent', 'N/D')}")
        c3.write(
            f"**Evidências:** "
            f"{result.get('evidence_count', len(safe_list(result.get('citations'))))}"
        )

        c1.write(
            f"**Documentos recuperados:** "
            f"{len(safe_list(result.get('retrieved')))}"
        )
        c2.write(
            f"**Chunks reranked:** "
            f"{len(safe_list(result.get('reranked')))}"
        )
        c3.write(
            f"**Latência:** "
            f"{result.get('latency_ms', 'N/D')} ms"
        )

        guard = result.get("guard", {})
        if isinstance(guard, dict):
            st.write(
                "**Guard Agent:** "
                + (
                    "🟢 Permitido"
                    if guard.get("allowed", True)
                    else "🔴 Bloqueado"
                )
            )

        if result.get("reason"):
            st.info(str(result["reason"]))

        if result.get("error"):
            st.error(str(result["error"]))


# ============================================================
# LOGIN
# ============================================================

user = get_current_user()

if not user:
    st.markdown(
        """
        <div style="max-width:520px;margin:7rem auto 0;text-align:center;">
            <div style="font-size:3.2rem;">⚖️</div>
            <h1>Assistente Jurídico IA</h1>
            <p style="color:#6c7890;">
                Inteligência artificial para documentos, riscos,
                pesquisas e análises jurídicas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login"):
        email = st.text_input("E-mail", "admin@demo.local")
        password = st.text_input("Senha", "admin123", type="password")
        submitted = st.form_submit_button(
            "Entrar",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            try:
                if authenticate(email, password):
                    st.rerun()
                else:
                    st.error("Credenciais inválidas.")
            except Exception as exc:
                st.error(f"Erro durante autenticação: {exc}")

    st.info("Demo: admin@demo.local / admin123")
    st.stop()


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">⚖️</div>
            <div class="title">ASSISTENTE JURÍDICO IA</div>
            <div class="sub">Inteligência que fortalece sua atuação</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    navigation = [
        ("PRINCIPAL", [
            ("⌂", "Dashboard"),
            ("▣", "Assistente IA"),
        ]),
        ("INTELIGÊNCIA", [
            ("⌕", "Pesquisas Jurídicas"),
            ("⚖", "Jurisprudência"),
            ("◈", "Análise de Risco"),
        ]),
        ("GESTÃO JURÍDICA", [
            ("□", "Documentos"),
            ("▤", "Processos"),
            ("▱", "Modelos e Petições"),
            ("◷", "Agenda e Prazos"),
            ("♙", "Clientes"),
        ]),
        ("SISTEMA", [
            ("⚙", "Configurações"),
            ("▣", "Auditoria"),
            ("?", "Ajuda"),
        ]),
    ]

    for group, items in navigation:
        st.caption(group)
        for icon, item in items:
            if st.button(
                f"{icon}  {item}",
                key=f"nav_{item}",
                use_container_width=True,
            ):
                st.session_state.page = item
                st.rerun()

    st.markdown(
        """
        <div class="sidebar-plan">
            <div style="color:#63d89c;font-weight:700;">● Plano Profissional</div>
            <div style="margin-top:.7rem;color:#a9bad3;font-size:.75rem;">
                Créditos utilizados
            </div>
            <div class="value">2.450 <span style="font-size:.75rem;color:#a9bad3;">/ 10.000</span></div>
            <div style="height:6px;background:#294667;border-radius:99px;margin-top:.6rem;">
                <div style="width:24%;height:6px;background:#35c99a;border-radius:99px;"></div>
            </div>
            <div style="text-align:right;color:#a9bad3;font-size:.7rem;margin-top:.25rem;">24%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"👤 {user.get('name', 'Usuário')}")
    st.caption(user.get("role", "user"))

    if st.button("Sair", use_container_width=True):
        logout()
        st.rerun()


page = st.session_state.page


# ============================================================
# TOPBAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div>
            <span class="brand-title">Assistente Jurídico IA</span>
            <span class="version-pill">V3.1 Profissional</span>
        </div>
        <div style="color:#6c7890;">🟢 IA Online &nbsp; · &nbsp; 🔔</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Olá, Dr. 👋</h1>
            <p>Como posso ajudar você hoje?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    metrics = [
        ("▣", "Conversas", "128", "+12% este mês"),
        ("□", "Documentos", "48", "+8% este mês"),
        ("⌕", "Pesquisas", "96", "+15% este mês"),
        ("◈", "Créditos", "2.450", "de 10.000"),
    ]

    cols = st.columns(4)
    for col, (icon, label, value, trend) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.25rem;">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-trend">{trend}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2.1, 1])

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Pergunte algo sobre seus documentos ou sobre direito...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        dash_q = st.text_input(
            "Pergunta jurídica",
            placeholder="Ex.: Quem é a CONTRATANTE?",
            label_visibility="collapsed",
            key="dashboard_question",
        )

        a, b, c = st.columns([1, 1, 1.3])
        with a:
            attach = st.button("📎 Anexar documentos", use_container_width=True)
        with b:
            research = st.button("⚖️ Pesquisa Jurídica", use_container_width=True)
        with c:
            risk = st.button("🛡️ Análise de Risco", use_container_width=True)

        if dash_q and st.button("➤ Perguntar", type="primary"):
            st.session_state.page = "Assistente IA"
            st.session_state.pending_question = dash_q
            st.rerun()

        if attach:
            st.session_state.page = "Documentos"
            st.rerun()
        if research:
            st.session_state.page = "Pesquisas Jurídicas"
            st.rerun()
        if risk:
            st.session_state.page = "Análise de Risco"
            st.rerun()

        st.markdown(
            """
            <div style="margin-top:.7rem;">
                <span class="question-chip">Quais são os riscos deste contrato?</span>
                <span class="question-chip">Resuma a petição inicial</span>
                <span class="question-chip">Qual o prazo para recurso?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📄 Documentos Recentes</div>',
            unsafe_allow_html=True,
        )

        docs = [
            ("Contrato de Prestação de Serviços.pdf", "Contrato · 2,4 MB", "Analisado"),
            ("Petição Inicial - Indenização.pdf", "Petição · 1,8 MB", "Analisado"),
            ("Procuração - Cliente.docx", "Procuração · 0,5 MB", "Analisado"),
            ("Sentença - Processo.pdf", "Sentença · 3,2 MB", "Pendente"),
        ]

        for name, meta, status in docs:
            with st.container(border=True):
                c1, c2, c3 = st.columns([0.1, 0.72, 0.18])
                with c1:
                    st.markdown("📄")
                with c2:
                    st.write(f"**{name}**")
                    st.caption(meta)
                with c3:
                    st.success(status) if status == "Analisado" else st.warning(status)

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">🛡️ Análise de Risco IA</div>
                <div style="font-size:3rem;font-weight:800;text-align:center;margin:1rem 0;">
                    18
                </div>
                <div style="text-align:center;color:#6c7890;">riscos analisados</div>
                <hr>
                <div>🔴 Alto risco <b style="float:right;">5</b></div>
                <div>🟠 Médio risco <b style="float:right;">7</b></div>
                <div>🟢 Baixo risco <b style="float:right;">6</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
                <div class="section-title">⚖️ Pesquisas Jurídicas Recentes</div>
                <p><b>Jurisprudência sobre danos morais</b><br><small>STJ · recente</small></p>
                <hr>
                <p><b>Prazo para interposição de recurso</b><br><small>STF · recente</small></p>
                <hr>
                <p><b>Responsabilidade civil do fornecedor</b><br><small>STJ · recente</small></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="section-title">🔎 Resumo da Última Análise</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns([0.8, 1.6, 1.4])
    with a:
        st.markdown(
            """
            <div class="risk-box">
                <div style="color:#6c7890;">Nível de risco</div>
                <div style="font-size:1.8rem;font-weight:800;color:#dc3545;">ALTO</div>
                <div>85/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.write("**Principais riscos identificados**")
        st.write("• Cláusula de rescisão unilateral")
        st.write("• Ausência de multa compensatória")
        st.write("• Prazo contratual desproporcional")
        st.write("• Limitação excessiva de responsabilidade")
    with c:
        st.write("**Recomendação**")
        st.write(
            "Revisar as cláusulas destacadas para equilíbrio "
            "contratual e mitigação dos riscos jurídicos."
        )

    st.markdown(
        '<div class="footer-note">Assistente Jurídico IA · RAG + Multiagentes + Evidências · V3.1</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ASSISTENTE IA
# ============================================================

elif page == "Assistente IA":

    st.markdown(
        """
        <div class="hero">
            <h1>🧑‍⚖️ Assistente Jurídico IA</h1>
            <p>Analise documentos, consulte sua base jurídica e obtenha respostas fundamentadas em evidências.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_question", None)
    q = st.chat_input(
        "Pergunte sobre seus documentos ou sobre direito..."
    )

    q = q or pending

    if q:
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner(
                "Analisando → Guard → RAG → Retriever → Reranker → Agente → LLM..."
            ):
                result = call_orchestrator(
                    query=q,
                    org_id=user.get("organization_id"),
                    mode="auto",
                    top_k=8,
                    rerank_k=5,
                )

            response = str(result.get("answer", "") or "").strip()

            if response:
                st.markdown(response)
            else:
                st.warning(
                    "A execução terminou sem uma resposta textual. "
                    "As evidências recuperadas permanecem disponíveis abaixo."
                )

            render_citations(result.get("citations"))
            render_diagnostic(result)

            if result.get("error"):
                with st.expander("⚠️ Detalhes técnicos"):
                    st.code(str(result["error"]))

            try:
                audit(
                    user,
                    "rag.ask",
                    "conversation",
                    None,
                    {
                        "query": q,
                        "agent": result.get("agent"),
                        "intent": result.get("intent"),
                        "retrieved": len(safe_list(result.get("retrieved"))),
                        "reranked": len(safe_list(result.get("reranked"))),
                        "evidence_count": result.get("evidence_count", 0),
                        "latency_ms": result.get("latency_ms", 0),
                    },
                )
            except Exception:
                pass

        st.session_state.messages.append(
            {"role": "assistant", "content": response or "Execução sem resposta textual."}
        )


# ============================================================
# DOCUMENTOS
# ============================================================

elif page == "Documentos":

    st.markdown(
        """
        <div class="hero">
            <h1>📄 Documentos</h1>
            <p>Centralize contratos, petições, procurações e demais documentos jurídicos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    up = st.file_uploader(
        "Adicionar documento",
        type=["pdf", "docx", "txt"],
    )

    use_ocr = st.checkbox("Usar OCR quando necessário", True)

    if up and st.button("🚀 Processar e indexar", type="primary"):
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
                        user.get("organization_id"),
                        use_ocr=use_ocr,
                    )
                )

                status.update(
                    label="Documento processado com sucesso",
                    state="complete",
                )

                st.success(
                    f"{result.get('chunks', 0)} chunks · "
                    f"{result.get('pages', 0)} páginas · "
                    f"OCR: {result.get('ocr_pages', 0)} páginas"
                )

            except Exception as exc:
                status.update(label="Falha no processamento", state="error")
                st.error(f"Erro: {exc}")
                st.code(traceback.format_exc())

    st.markdown("### Documentos disponíveis")

    try:
        documents = list_documents(user.get("organization_id"))

        if not documents:
            st.info("Nenhum documento encontrado.")

        for d in documents:
            if not isinstance(d, dict):
                continue

            with st.container(border=True):
                c1, c2, c3 = st.columns([0.08, 0.72, 0.2])
                with c1:
                    st.markdown("📄")
                with c2:
                    st.write(f"**{d.get('name', 'Documento')}**")
                    st.caption(
                        f"{d.get('type', 'N/D')} · "
                        f"{d.get('status', 'N/D')} · "
                        f"{d.get('pages', 0)} páginas · "
                        f"{d.get('chunks', 0)} chunks"
                    )
                with c3:
                    st.write(d.get("status", "N/D"))

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
                            query=(
                                "Analise os principais riscos jurídicos, "
                                "evidências, lacunas e recomendações. "
                                "Não invente fatos e fundamente cada risco "
                                "nas evidências disponíveis."
                            ),
                            org_id=user.get("organization_id"),
                            top_k=8,
                            rerank_k=5,
                            extra_context=text,
                        )
                    )

                    answer = str(result.get("answer", "") or "").strip()

                    if answer:
                        st.markdown(answer)
                    else:
                        st.warning(
                            "O agente terminou sem resposta textual. "
                            "Verifique as evidências e o diagnóstico."
                        )

                    render_citations(result.get("citations"))
                    render_diagnostic(result)

                except Exception as exc:
                    st.error(f"Erro na análise de risco: {exc}")
                    st.code(traceback.format_exc())


# ============================================================
# BASE VETORIAL
# ============================================================

elif page == "Base Vetorial":

    st.markdown(
        """
        <div class="hero">
            <h1>🔎 Base Vetorial</h1>
            <p>Explore os documentos recuperados pelo Retriever e pelo Reranker.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q = st.text_input("Consulta")

    if q:
        try:
            result = safe_dict(
                retrieve_and_rerank(
                    q,
                    user.get("organization_id"),
                    top_k=10,
                    rerank_k=5,
                )
            )

            retrieved = safe_list(result.get("retrieved"))
            reranked = safe_list(result.get("reranked"))

            st.info(
                f"Retriever: {len(retrieved)} resultados · "
                f"Reranker: {len(reranked)} resultados"
            )

            for i, x in enumerate(reranked, 1):
                if not isinstance(x, dict):
                    continue

                with st.container(border=True):
                    st.write(
                        f"**[{i}] {x.get('document', 'Documento')} · "
                        f"página {x.get('page', 'N/D')}**"
                    )
                    st.caption(
                        f"chunk={x.get('chunk_id', 'N/D')} · "
                        f"retriever={float(x.get('retriever_score', 0) or 0):.4f} · "
                        f"reranker={float(x.get('reranker_score', 0) or 0):.4f}"
                    )
                    st.write(x.get("content", ""))

        except Exception as exc:
            st.error(f"Erro na busca vetorial: {exc}")


# ============================================================
# AVALIAÇÃO
# ============================================================

elif page == "Avaliação RAG":

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Avaliação da Resposta</h1>
            <p>Context Relevance · Citation Coverage · Groundedness</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area("Pergunta")
    answer = st.text_area("Resposta da IA")

    if st.button("🧪 Avaliar", type="primary"):
        if not question.strip():
            st.warning("Informe a pergunta.")
        elif not answer.strip():
            st.warning("Informe a resposta.")
        else:
            try:
                rag_result = safe_dict(
                    rag_answer(
                        question,
                        user.get("organization_id"),
                        top_k=8,
                        rerank_k=5,
                        generate_answer_flag=False,
                    )
                )

                score = evaluate_answer(
                    question,
                    answer,
                    safe_list(rag_result.get("reranked")),
                    safe_list(rag_result.get("citations")),
                )

                a, b, c, d = st.columns(4)
                a.metric("Context relevance", f"{score['context_relevance']:.2f}")
                b.metric("Citation coverage", f"{score['citation_coverage']:.2f}")
                c.metric("Groundedness", f"{score['groundedness']:.2f}")
                d.metric("Overall", f"{score['overall']:.2f}")

                st.markdown(
                    f"### Qualidade: **{score.get('quality', 'N/D')}**"
                )

                if score.get("recommendations"):
                    st.markdown("### 💡 Recomendações")
                    for recommendation in score["recommendations"]:
                        st.info(recommendation)

                with st.expander("JSON completo"):
                    st.json(score)

            except Exception as exc:
                st.error(f"Erro na avaliação: {exc}")


# ============================================================
# PROCESSOS
# ============================================================

elif page == "Processos":

    st.markdown(
        """
        <div class="hero">
            <h1>⚖️ Processos</h1>
            <p>Controle processos, prioridades, clientes e status.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("new_case"):
        title = st.text_input("Título")
        client = st.text_input("Cliente")
        category = st.selectbox(
            "Categoria",
            ["Cível", "Trabalhista", "Contratos", "Tributário", "Previdenciário", "Outros"],
        )
        priority = st.selectbox("Prioridade", ["Baixa", "Média", "Alta"])
        submitted = st.form_submit_button("Cadastrar", type="primary")

        if submitted:
            if not title.strip():
                st.warning("Informe o título do processo.")
            else:
                try:
                    create_case(
                        user.get("organization_id"),
                        title,
                        client,
                        category,
                        priority,
                    )
                    st.success("Processo criado.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erro ao criar processo: {exc}")

    try:
        cases = list_cases(user.get("organization_id"))
        for c in cases:
            if not isinstance(c, dict):
                continue
            with st.container(border=True):
                st.write(f"**{c.get('title', 'Processo')}**")
                st.caption(
                    f"{c.get('client', 'N/D')} · "
                    f"{c.get('category', 'N/D')} · "
                    f"{c.get('priority', 'N/D')} · "
                    f"{c.get('status', 'N/D')}"
                )
    except Exception as exc:
        st.error(f"Erro ao carregar processos: {exc}")


# ============================================================
# PESQUISAS JURÍDICAS / JURISPRUDÊNCIA
# ============================================================

elif page in ("Pesquisas Jurídicas", "Jurisprudência"):

    title = "🔎 Pesquisas Jurídicas" if page == "Pesquisas Jurídicas" else "⚖️ Jurisprudência"

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>Área preparada para consultas jurídicas com fontes e evidências.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    q = st.text_input(
        "O que deseja pesquisar?",
        placeholder="Ex.: responsabilidade civil por falha na prestação de serviço",
    )

    if st.button("Pesquisar", type="primary"):
        if not q.strip():
            st.warning("Digite uma consulta.")
        else:
            st.info(
                "A interface de pesquisa está preparada. "
                "Para pesquisa jurídica externa com fontes atuais, "
                "o conector de pesquisa precisa estar ligado ao seu backend."
            )

    st.markdown("### Pesquisas recentes")
    for item in [
        "Jurisprudência sobre danos morais",
        "Prazo para interposição de recurso",
        "Responsabilidade civil do fornecedor",
    ]:
        with st.container(border=True):
            st.write(f"**{item}**")
            st.caption("Pesquisa jurídica · fonte a consultar")


# ============================================================
# MODELOS / AGENDA / CLIENTES / AJUDA
# ============================================================

elif page == "Modelos e Petições":

    st.markdown(
        """
        <div class="hero">
            <h1>📝 Modelos e Petições</h1>
            <p>Biblioteca para organizar modelos jurídicos e documentos reutilizáveis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Módulo preparado para integração com sua biblioteca de modelos.")


elif page == "Agenda e Prazos":

    st.markdown(
        """
        <div class="hero">
            <h1>◷ Agenda e Prazos</h1>
            <p>Organize compromissos e prazos processuais.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Módulo de agenda preparado para integração com calendário e prazos.")


elif page == "Clientes":

    st.markdown(
        """
        <div class="hero">
            <h1>♙ Clientes</h1>
            <p>Central de clientes e relacionamento jurídico.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Módulo de clientes preparado para integração com o cadastro do sistema.")


# ============================================================
# AUDITORIA
# ============================================================

elif page == "Auditoria":

    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ Auditoria</h1>
            <p>Histórico de eventos e ações realizadas no tenant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from db import get_connection

        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT action, entity_type, entity_id, created_at, metadata
                FROM audit_logs
                WHERE organization_id=?
                ORDER BY id DESC
                LIMIT 100
                """,
                (user.get("organization_id"),),
            ).fetchall()

        if not rows:
            st.info("Nenhum evento encontrado.")

        for r in rows:
            st.write(
                f"`{r['created_at']}` · **{r['action']}** · "
                f"{r['entity_type']}#{r['entity_id']} · {r['metadata']}"
            )

    except Exception as exc:
        st.error(f"Erro ao carregar auditoria: {exc}")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

elif page == "Configurações":

    st.markdown(
        """
        <div class="hero">
            <h1>⚙️ Configurações</h1>
            <p>Controle LLM, RAG, segurança, agentes e parâmetros da plataforma.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🤖 Inteligência Artificial")
        llm = st.selectbox("LLM", ["Modo Demo", "Gemini", "OpenAI"])
        top_k = st.number_input("Retriever Top-K", 1, 50, 8)
        rerank_k = st.number_input("Reranker Top-K", 1, 20, 5)

    with c2:
        st.markdown("### 🛡️ Segurança")
        citations_required = st.checkbox("Citações obrigatórias", True)
        evaluation_enabled = st.checkbox("Evaluation habilitado", True)
        guard_enabled = st.checkbox("Guard Agent habilitado", True)

    st.markdown("### 🧠 Agentes")
    agents = [
        "Agente Jurídico",
        "Agente de Risco",
        "Agente de Resumo",
        "Agente Geral",
        "RAG",
        "Citações",
        "Evaluation",
        "Guard Agent",
    ]

    cols = st.columns(4)
    for i, name in enumerate(agents):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="card">
                    <b>{name}</b><br>
                    <span class="status-ok">● Ativo</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.success(
        f"Configuração atual: LLM={llm} · Top-K={top_k} · Reranker={rerank_k}"
    )


# ============================================================
# AJUDA
# ============================================================

elif page == "Ajuda":

    st.markdown(
        """
        <div class="hero">
            <h1>❓ Ajuda</h1>
            <p>Como utilizar o Assistente Jurídico IA.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Fluxo recomendado

        1. Envie o documento em **Documentos**.
        2. Aguarde a ingestão e indexação.
        3. Vá para **Assistente IA**.
        4. Faça perguntas sobre o documento.
        5. Confira **evidências, páginas e citações**.
        6. Use **Análise de Risco** para contratos.
        7. Use **Avaliação RAG** para medir a qualidade da resposta.
        """
    )


else:
    st.session_state.page = "Dashboard"
    st.rerun()
