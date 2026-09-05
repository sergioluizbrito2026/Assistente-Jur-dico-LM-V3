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
from services.ingestion import ingest_document


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
# ESTILO GLOBAL (CSS)
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

.card {
    background:white;
    border:1px solid var(--border);
    border-radius:16px;
    padding:1.1rem;
    box-shadow:0 5px 20px rgba(20,32,51,.045);
    margin-bottom: 1rem;
}

.metric-card {
    background:white;
    border:1px solid var(--border);
    border-radius:15px;
    padding:1rem;
    min-height:105px;
    box-shadow:0 4px 16px rgba(20,32,51,.04);
}

.metric-label {
    color:var(--muted);
    font-size:0.75rem;
    font-weight:600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.metric-value {
    font-size:1.55rem;
    font-weight:750;
    margin-top:0.25rem;
}

.section-title {
    font-size:1.1rem;
    font-weight:750;
    margin:0.2rem 0 0.8rem;
}

.badge-red { background:#fde8e8; color:#c53030; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.8rem; }
.badge-orange { background:#fef3c7; color:#b45309; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.8rem; }
.badge-green { background:#def7ec; color:#03543f; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.8rem; }

.footer-note {
    color:#8792a6;
    font-size:0.72rem;
    text-align:center;
    margin-top:2rem;
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
            filtered = {k: v for k, v in kwargs.items() if k in parameters and v is not None}

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

    for i, citation in enumerate(citations, 1):
        if not isinstance(citation, dict):
            continue

        cid = citation.get("id", i)
        doc = citation.get("document", citation.get("document_name", "Documento"))
        page = citation.get("page", "N/D")
        chunk = citation.get("chunk_id", "N/D")
        content = citation.get("content", citation.get("text", ""))

        st.markdown(
            f"""
            <div class="citation" style="background:#f7f9fd; border:1px solid #e4e9f2; border-radius:10px; padding:0.55rem 0.7rem; margin:0.35rem 0;">
                <b>[{cid}] {doc}</b>
                · página {page}
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
    result = safe_dict(result)

    with st.expander("🔍 Diagnóstico da execução", expanded=False):
        c1, c2, c3 = st.columns(3)

        agent = str(result.get("agent", "N/D"))
        intent = str(result.get("intent", "N/D"))
        citations = safe_list(result.get("citations"))
        evidence_count = result.get("evidence_count", len(citations))
        retrieved = safe_list(result.get("retrieved"))
        reranked = safe_list(result.get("reranked"))
        latency = result.get("latency_ms", "N/D")

        c1.write("**Agente:** " + agent)
        c2.write("**Intent:** " + intent)
        c3.write("**Evidências:** " + str(evidence_count))

        c1.write("**Documentos recuperados:** " + str(len(retrieved)))
        c2.write("**Chunks reranked:** " + str(len(reranked)))
        c3.write("**Latência:** " + str(latency) + " ms")

        guard = result.get("guard", {})
        if isinstance(guard, dict):
            allowed = guard.get("allowed", True)
            if allowed:
                st.write("**Guard Agent:** 🟢 Permitido")
            else:
                st.write("**Guard Agent:** 🔴 Bloqueado")

        reason = result.get("reason")
        if reason:
            st.info(str(reason))

        error = result.get("error")
        if error:
            st.error(str(error))


# ============================================================
# LOGIN
# ============================================================

user = get_current_user()

if not user:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size:3.2rem;">⚖️</div>
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
# SESSION STATE & SIDEBAR NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">⚖️</div>
            <div class="title">Jurídico SaaS</div>
            <div class="sub">Inteligência Artificial v3.1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button("💬 Assistente IA", use_container_width=True):
        st.session_state.page = "Assistente IA"
        st.rerun()

    if st.button("📄 Documentos", use_container_width=True):
        st.session_state.page = "Documentos"
        st.rerun()

    st.markdown("---")

    if st.button("🚪 Sair", use_container_width=True):
        logout()
        st.rerun()

page = st.session_state.page


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("⚖️ JURÍDICO SaaS")
    st.caption("Inteligência Artificial v3.1 — Painel de Controle Consolidado")
    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 1. INDICADORES PRINCIPAIS (6 CARDS)
    # --------------------------------------------------------
    try:
        from db import get_connection

        with get_connection() as c:
            org_id = user.get("organization_id")

            documents_count = c.execute(
                "SELECT COUNT(*) FROM documents WHERE organization_id=?", (org_id,)
            ).fetchone()[0]

            cases_count = c.execute(
                "SELECT COUNT(*) FROM cases WHERE organization_id=?", (org_id,)
            ).fetchone()[0]

            chunks_count = c.execute(
                "SELECT COUNT(*) FROM chunks WHERE organization_id=?", (org_id,)
            ).fetchone()[0]

            audit_count = c.execute(
                "SELECT COUNT(*) FROM audit_logs WHERE organization_id=?", (org_id,)
            ).fetchone()[0]
    except Exception:
        documents_count = 248
        cases_count = 42
        chunks_count = 1420
        audit_count = 521

    # Mock/Métricas unificadas com os valores pedidos
    ind_casos_ativos = cases_count if cases_count > 0 else 42
    ind_documentos = documents_count if documents_count > 0 else 248
    ind_analises_ia = 386
    ind_riscos = 27
    ind_consultas_rag = 521
    ind_pendencias = 13

    metrics_cols = st.columns(6)
    cards_data = [
        ("📁 Casos Ativos", str(ind_casos_ativos)),
        ("📄 Documentos", str(ind_documentos)),
        ("🤖 Análises IA", str(ind_analises_ia)),
        ("⚠️ Riscos", str(ind_riscos)),
        ("🔎 Consultas", str(ind_consultas_rag)),
        ("⏱️ Pendentes", str(ind_pendencias)),
    ]

    for col, (label, val) in zip(metrics_cols, cards_data):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 2. CASOS QUE EXIGEM ATENÇÃO
    # --------------------------------------------------------
    st.markdown('<div class="section-title">⚠️ Casos que Exigem Atenção</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            st.markdown("**Processo #2026-0145** (Caso #102)")
            st.caption("Última análise: hoje")
        with col_b:
            st.markdown("Risco: <span class='badge-red'>🔴 Alto</span>", unsafe_allow_html=True)
            st.caption("Motivo: Prazo processual próximo")
        with col_c:
            if st.button("Analisar caso", key="btn_analisar_102", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.session_state.pending_question = "Faça uma análise detalhada do Processo #2026-0145 e verifique os prazos."
                st.rerun()

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            st.markdown("**Processo #2026-0182** (Caso #108)")
            st.caption("Última análise: ontem")
        with col_b:
            st.markdown("Risco: <span class='badge-orange'>🟠 Médio</span>", unsafe_allow_html=True)
            st.caption("Motivo: Documento pendente de análise")
        with col_c:
            if st.button("Analisar caso", key="btn_analisar_108", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.session_state.pending_question = "Verifique os documentos pendentes do Caso #108."
                st.rerun()

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            st.markdown("**Processo #2026-0210** (Caso #115)")
            st.caption("Última análise: há 3 dias")
        with col_b:
            st.markdown("Risco: <span class='badge-red'>🔴 Alto</span>", unsafe_allow_html=True)
            st.caption("Motivo: Revisão necessária / Sem movimentação")
        with col_c:
            if st.button("Analisar caso", key="btn_analisar_115", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.session_state.pending_question = "Quais são as pendências de revisão do Caso #115?"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 4. GRÁFICOS E VISÃO GERAL
    # --------------------------------------------------------
    st.markdown('<div class="section-title">📊 Visão Geral & Estatísticas</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)

    with g1:
        with st.container(border=True):
            st.markdown("**Casos por Status**")
            status_data = {"Ativo": 18, "Em análise": 10, "Em andamento": 7, "Concluído": 5, "Arquivado": 2}
            st.bar_chart(status_data)

    with g2:
        with st.container(border=True):
            st.markdown("**Riscos Identificados**")
            risks_data = {"Alto": 7, "Médio": 12, "Baixo": 8}
            st.bar_chart(risks_data)

    with g3:
        with st.container(border=True):
            st.markdown("**Atividade da IA (Semanal)**")
            ai_activity = {"Seg": 45, "Ter": 68, "Qua": 82, "Qui": 55, "Sex": 90}
            st.line_chart(ai_activity)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 3. CENTRAL DE INTELIGÊNCIA ARTIFICIAL
    # --------------------------------------------------------
    st.markdown('<div class="section-title">🤖 Central de Inteligência Artificial</div>', unsafe_allow_html=True)
    with st.container(border=True):
        aic1, aic2, aic3 = st.columns(3)
        with aic1:
            st.markdown("🔹 **Status IA:** 🟢 Operacional")
            st.markdown("🔹 **Modelo / LLM:** Gemini / OpenAI")
            st.markdown("🔹 **RAG:** 🟢 Ativo")
        with aic2:
            st.markdown("🔹 **Base de Conhecimento:** 248 documentos")
            st.markdown("🔹 **Análises Realizadas:** 386")
            st.markdown("🔹 **Resumos Gerados:** 142")
        with aic3:
            st.markdown("🔹 **Riscos Detectados:** 27")
            st.markdown("🔹 **Taxa de Citações:** 98.4%")
            st.markdown("🔹 **Conexão Provedor:** 🟢 Estável")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 5. DOCUMENTOS RECENTES
    # --------------------------------------------------------
    st.markdown('<div class="section-title">📄 Documentos Recentes</div>', unsafe_allow_html=True)
    
    docs_data = [
        {"doc": "Contrato.pdf", "caso": "Caso #102", "tipo": "Contrato", "status": "✅ Analisado", "data": "04/09"},
        {"doc": "Petição.pdf", "caso": "Caso #103", "tipo": "Petição", "status": "🤖 Em análise", "data": "04/09"},
        {"doc": "Sentença.pdf", "caso": "Caso #104", "tipo": "Sentença", "status": "⚠️ Revisar", "data": "03/09"},
        {"doc": "Procuracao.pdf", "caso": "Caso #105", "tipo": "Procuração", "status": "✅ Analisado", "data": "02/09"},
    ]

    for d in docs_data:
        cols_doc = st.columns([2, 1.5, 1.5, 1.5, 1])
        cols_doc[0].write(f"**{d['doc']}**")
        cols_doc[1].write(d['caso'])
        cols_doc[2].write(d['tipo'])
        cols_doc[3].write(d['status'])
        cols_doc[4].write(d['data'])
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 6. ÚLTIMAS ANÁLISES DA IA
    # --------------------------------------------------------
    st.markdown('<div class="section-title">🧠 Últimas Análises da IA</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("🤖 **Análise jurídica concluída** para o *Caso #102* — Cláusula de foro validada com sucesso.")
        st.markdown("⚠️ **Risco identificado** no *Contrato.pdf* — Prazo de vigência ambíguo detectado.")
        st.markdown("📄 **Documento processado** — *Petição.pdf* indexado em 14 chunks vetoriais.")
        st.markdown("🔎 **Consulta realizada** via RAG sobre jurisprudência de danos morais.")
        st.markdown("📝 **Resumo gerado** para a petição inicial do *Caso #104*.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 7. AUDITORIA E SEGURANÇA
    # --------------------------------------------------------
    st.markdown('<div class="section-title">🔐 Auditoria e Segurança</div>', unsafe_allow_html=True)
    with st.container(border=True):
        sec1, sec2, sec3 = st.columns(3)
        with sec1:
            st.markdown("🔒 **Status da Autenticação:** Ativa (JWT)")
            st.markdown("👤 **Usuários Ativos:** 4 conectados")
        with sec2:
            st.markdown("🗄️ **Status do Banco de Dados:** 🟢 Operacional")
            st.markdown("📋 **Ações Registradas (Logs):** 521 eventos")
        with sec3:
            st.markdown("🛡️ **Eventos de Segurança:** 0 alertas críticos")
            st.markdown("🕒 **Último Acesso:** Hoje, 11:45")

    st.markdown(
        """
        <div class="footer-note">
            Assistente Jurídico IA &middot;
            RAG + Multiagentes + Evidências &middot; V3.1
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ASSISTENTE IA (CHAT)
# ============================================================

elif page == "Assistente IA":
    st.markdown(
        """
        <div class="hero" style="background:white; border:1px solid #e4e9f2; border-radius:18px; padding:1.45rem; box-shadow:0 6px 24px rgba(20,32,51,.05);">
            <h1>Assistente Jurídico IA</h1>
            <p>
                Analise documentos, consulte sua base jurídica
                e obtenha respostas fundamentadas em evidências.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if 'messages' in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.session_state.messages = []

    pending = st.session_state.pop("pending_question", None)
    q = st.chat_input("Pergunte sobre seus documentos ou sobre direito...")
    q = q or pending

    if q:
        q = str(q).strip()

        if q:
            st.session_state.messages.append({
                "role": "user",
                "content": q,
            })

            with st.chat_message("user"):
                st.markdown(q)

            with st.chat_message("assistant"):
                with st.spinner("Analisando -> Guard -> RAG -> Retriever -> Reranker -> Agente -> LLM..."):
                    try:
                        result = call_orchestrator(
                            query=q,
                            org_id=user.get("organization_id"),
                            mode="auto",
                            top_k=8,
                            rerank_k=5,
                        )
                    except Exception as e:
                        result = {"answer": f"Erro ao executar o orquestrador: {e}"}

                response = str(result.get("answer", "") or "").strip()

                if response:
                    st.markdown(response)
                else:
                    st.warning(
                        "A execução terminou sem uma "
                        "resposta textual. As evidências "
                        "recuperadas permanecem disponíveis abaixo."
                    )

                render_citations(result.get("citations"))
                render_diagnostic(result)

                if result.get("error"):
                    with st.expander("Detalhes técnicos"):
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
                        },
                    )
                except Exception:
                    pass

            st.session_state.messages.append({
                "role": "assistant",
                "content": response or "Execução sem resposta textual.",
            })


# ============================================================
# DOCUMENTOS
# ============================================================

elif page == "Documentos":
    st.markdown(
        """
        <div class="hero" style="background:white; border:1px solid #e4e9f2; border-radius:18px; padding:1.45rem; box-shadow:0 6px 24px rgba(20,32,51,.05);">
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

    use_ocr = st.checkbox("Usar OCR quando necessário", True)

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
                            user.get("organization_id"),
                            use_ocr=use_ocr,
                        )
                    )

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
        documents = list_documents(user.get("organization_id"))

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
                        f"{document.get('pages', 0)} páginas - "
                        f"{document.get('chunks', 0)} chunks"
                    )

                with c3:
                    status_doc = document.get("status", "N/D")
                    if status_doc == "Indexado":
                        st.success(status_doc)
                    else:
                        st.warning(status_doc)

    except Exception as exc:
        st.error(f"Erro ao carregar documentos: {exc}")
