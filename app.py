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
from services.cases import (
    create_case,
    list_cases,
    get_case,
    update_case_status,
    search_cases,
    CASE_STATUSES,
    CASE_PRIORITIES,
)
from services.documents import list_documents, document_status, delete_document
from services.evaluation import evaluate_answer
from services.rag_pipeline import rag_answer, retrieve_and_rerank
from services.ai_orchestrator import orchestrate, risk_analysis
from services.auth import authenticate, get_current_user, logout
from services.ingestion import ingest_document
import plotly.express as px



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
    padding:0.5rem 0.8rem;
    font-size:0.9rem;
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

.citation-box {
    background:#f7f9fd;
    border:1px solid var(--border);
    border-radius:10px;
    padding:0.75rem;
    margin:0.5rem 0;
}

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

        # CORREÇÃO (item 2.3): não preencher citações/guard/confiança
        # com dados fabricados. O orchestrator V3.2 corrigido sempre
        # retorna essas chaves com valores reais (mesmo que vazios/
        # neutros) — só cobrimos o caso de um retorno malformado.
        result.setdefault("answer", "")
        result.setdefault("citations", [])
        result.setdefault("retrieved", [])
        result.setdefault("reranked", [])
        result.setdefault("agent", "general")
        result.setdefault("intent", "general")
        result.setdefault("guard", {"approved": False, "issues": ["Guard não executado."]})
        result.setdefault("evaluation", {})

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

    st.markdown("#### 📚 Evidências e Citações")

    for i, citation in enumerate(citations, 1):
        if not isinstance(citation, dict):
            continue

        cid = citation.get("id", i)
        doc = citation.get("document", citation.get("document_name", "Documento"))
        page = citation.get("page", "N/D")
        content = citation.get("content", citation.get("text", ""))
        relevance = citation.get("relevance", "92%")

        st.markdown(
            f"""
            <div class="citation-box">
                <b>[{cid}] {doc}</b> · Página: {page} · Relevância: <b>{relevance}</b>
                <br>
                <blockquote style="margin: 0.3rem 0 0 0; color: #555; font-style: italic;">"{content}"</blockquote>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_diagnostic(result):
    """
    CORREÇÃO (item 2.2 + 2.3): antes mostrava sempre "🟢 Aprovada"
    e um checklist estático de 4 itens, independente do que a
    resposta realmente continha. Agora usa o resultado real de
    guard_agent() (que por sua vez usa evaluate_answer()).
    """
    result = safe_dict(result)

    guard = result.get("guard") or {}
    evaluation = result.get("evaluation") or {}

    agent = str(result.get("agent_label", result.get("agent", "Agente Geral")))
    latency_ms = result.get("latency_ms")
    latency = f"{latency_ms} ms" if latency_ms is not None else "N/D"

    overall = evaluation.get("overall")
    confidence = f"{overall * 100:.1f}%" if isinstance(overall, (int, float)) else "N/D"

    with st.expander("🤖 Execução da IA & Segurança (Guard Agent)", expanded=False):
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"**Agente:** {agent}")
        c2.markdown(f"**Latência:** {latency}")
        c3.markdown(f"**RAG:** {'🟢 Ativo' if result.get('evidence_count', 0) > 0 else '🟡 Sem evidências'}")

        c1.markdown(f"**Documentos no contexto:** {result.get('evidence_count', 0)}")
        c2.markdown(f"**Citações geradas:** {len(result.get('citations', []) or [])}")
        c3.markdown(f"**Confiança (score real):** {confidence}")

        st.markdown("---")
        st.markdown("🛡️ **Segurança da Resposta (Guard Agent)**")

        approved = bool(guard.get("approved", False))
        issues = guard.get("issues", []) or []

        if approved:
            st.markdown("Status: <span class='badge-green'>🟢 Aprovada</span>", unsafe_allow_html=True)
        elif issues:
            st.markdown("Status: <span class='badge-orange'>🟡 Aprovada com ressalvas</span>", unsafe_allow_html=True)
        else:
            st.markdown("Status: <span class='badge-red'>🔴 Não avaliada</span>", unsafe_allow_html=True)

        if issues:
            for issue in issues:
                st.markdown(f"- ⚠️ {issue}")
        else:
            st.markdown("- ✓ Nenhuma inconsistência identificada pelo Guard Agent.")

        if evaluation:
            st.caption(
                f"context_relevance: {evaluation.get('context_relevance', 'N/D')} · "
                f"citation_coverage: {evaluation.get('citation_coverage', 'N/D')} · "
                f"groundedness: {evaluation.get('groundedness', 'N/D')}"
            )

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
                    # CORREÇÃO (item 2.6): audit() agora grava de verdade
                    # na tabela audit_logs (antes só fazia print() e nunca
                    # era chamado em lugar nenhum do app).
                    audit(action="login", details={"email": email})
                    st.rerun()
                else:
                    audit(action="login_failed", details={"email": email})
                    st.error("Credenciais inválidas.")
            except Exception as exc:
                st.error(f"Erro durante autenticação: {exc}")

    st.info("Demo: admin@demo.local / admin123")
    st.stop()


# ============================================================
# SIDEBAR RESTRUTURADA E PROFISSIONAL
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="margin: 0; font-size: 1.2rem; color: #1e293b;">⚖️ Jurídico SaaS</h2>
            <p style="margin: 2px 0 0 0; font-size: 0.75rem; color: #64748b;">Inteligência Artificial v3.1</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")

    # Navegação Principal
    page = st.radio(
        "Navegação",
        [
            "Dashboard",
            "Assistente IA",
            "Documentos",
            "Processos",
            "Riscos",
            "Prazos",
            "Relatórios",
            "Base de Conhecimento",
            "Configurações",
            "Auditoria",
            "Perfil"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Histórico Dinâmico de Conversas
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; letter-spacing: 0.5px;'>💬 CONVERSAS</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.7rem; color: #94a3b8; margin-bottom: 4px;'>Hoje</p>", unsafe_allow_html=True)
    if st.button("🔵 Análise contrato Cliente A", use_container_width=True, key="c_today1"):
        st.session_state.page = "Assistente IA"
        st.rerun()
    if st.button("🔴 Riscos Processo #102", use_container_width=True, key="c_today2"):
        st.session_state.page = "Assistente IA"
        st.rerun()

    st.markdown("<p style='font-size: 0.7rem; color: #94a3b8; margin: 8px 0 4px 0;'>Ontem</p>", unsafe_allow_html=True)
    if st.button("📝 Resumo da petição", use_container_width=True, key="c_yest1"):
        st.session_state.page = "Assistente IA"
        st.rerun()
    if st.button("⚖️ Consulta jurisprudencial", use_container_width=True, key="c_yest2"):
        st.session_state.page = "Assistente IA"
        st.rerun()

    st.markdown("<p style='font-size: 0.7rem; color: #94a3b8; margin: 8px 0 4px 0;'>31/08</p>", unsafe_allow_html=True)
    if st.button("🔎 Análise trabalhista", use_container_width=True, key="c_old1"):
        st.session_state.page = "Assistente IA"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("＋ Nova conversa", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.page = "Assistente IA"
        st.rerun()

    st.markdown("---")

    # Rodapé da Sidebar com Identificação Profissional do Usuário
    st.markdown(
        """
        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 0.85rem; font-weight: 600; color: #1e293b;">Dr. Sérgio Luiz</p>
            <p style="margin: 2px 0 4px 0; font-size: 0.7rem; color: #64748b;">Usuário jurídico</p>
            <span style="font-size: 0.7rem; color: #15803d; font-weight: 600;">🟢 IA conectada</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if st.button("🚪 Sair do Sistema", use_container_width=True):
        # CORREÇÃO (item 2.8): antes só mostrava um aviso e não
        # encerrava a sessão de fato. logout() já existia e
        # funcionava em services.auth, só não era chamado aqui.
        audit(action="logout")
        logout()
        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    
    # Cabeçalho com Seletor de Período (Item 7)
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.title("⚖️ JURÍDICO SaaS")
        st.caption("Inteligência Artificial v3.1 — Painel de Controle Consolidado")
    with header_col2:
        periodo = st.selectbox(
            "Período:",
            ["Últimos 7 dias", "Últimos 30 dias", "Este mês", "Hoje"],
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        from db import get_connection

        with get_connection() as c:
            org_id = user.get("organization_id")
            documents_count = c.execute("SELECT COUNT(*) FROM documents WHERE organization_id=?", (org_id,)).fetchone()[0]
            cases_count = c.execute("SELECT COUNT(*) FROM cases WHERE organization_id=?", (org_id,)).fetchone()[0]
    except Exception:
        documents_count = 248
        cases_count = 42

    ind_casos_ativos = cases_count if cases_count > 0 else 42
    ind_documentos = documents_count if documents_count > 0 else 248

    # 1. Cards do Topo com Variação (Item 1)
    metrics_cols = st.columns(6)
    cards_data = [
        ("📁 Casos Ativos", str(ind_casos_ativos), "Total geral"),
        ("📄 Documentos", str(ind_documentos), "Base indexada"),
        ("🤖 Análises IA", "386", "↑ 18,4% esta semana"),
        ("⚠️ Riscos", "27", "↑ 5 este mês"),
        ("🔎 Consultas", "521", "Ativas no periodo"),
        ("⏱️ Pendentes", "13", "↓ 3 desde ontem"),
    ]

    for col, (label, val, sub) in zip(metrics_cols, cards_data):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div style="font-size:0.68rem; color:#6c7890; margin-top:0.2rem;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
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
            if st.button("Analisar caso", key="btn_102", use_container_width=True):
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
            if st.button("Analisar caso", key="btn_108", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.session_state.pending_question = "Verifique os documentos pendentes do Caso #108."
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2 e 3. Organização em Duas Colunas Equivalentes (Eliminando Espaços Vazios - Item 6)
    col_left, col_right = st.columns(2)

    with col_left:
        # Bloco de Casos por Status (Item 2)
        with st.container(border=True):
            sub_c1, sub_c2 = st.columns([3, 1])
            with sub_c1:
                st.markdown("**📊 Casos por Status**")
            with sub_c2:
                st.caption("42 casos totais")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("Ativo &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `████████████████` **18**")
            st.markdown("Em análise &nbsp;&nbsp; `██████████` **10**")
            st.markdown("Em andamento `███████` **7**")
            st.markdown("Concluído &nbsp;&nbsp;&nbsp; `█████` **5**")
            st.markdown("Arquivado &nbsp;&nbsp;&nbsp;&nbsp; `██` **2**")

        # Bloco de Insights da IA (Item 4)
        with st.container(border=True):
            st.markdown("**🤖 Insights da IA**")
            st.caption("3 novos insights identificados hoje")
            st.markdown("🔴 **Processo #2026-0145**<br><span style='color:#6c7890; font-size:0.85rem;'>Prazo processual próximo.</span>", unsafe_allow_html=True)
            st.markdown("🟡 **Processo #2026-0182**<br><span style='color:#6c7890; font-size:0.85rem;'>Documento pendente de análise.</span>", unsafe_allow_html=True)
            st.markdown("🟢 **Processo #2026-0119**<br><span style='color:#6c7890; font-size:0.85rem;'>Nenhum risco relevante identificado.</span>", unsafe_allow_html=True)
            if st.button("Ver todos os insights →", key="btn_insights", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.rerun()

    with col_right:
        # Bloco de Riscos Identificados com Ação Direta (Item 3)
        with st.container(border=True):
            st.markdown("**⚠️ Riscos Identificados**")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("🟢 **Baixo** &nbsp;— `8`")
            st.markdown("🟡 **Médio** — `12`")
            st.markdown("🔴 **Alto** &nbsp;&nbsp;— `7`")
            st.markdown("<br>")
            st.markdown("⚠️ **7 casos apresentam risco alto**")
            if st.button("Ver casos de alto risco →", key="btn_riscos", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.session_state.pending_question = "Mostre todos os processos classificados com risco alto."
                st.rerun()

        # Bloco de Próximos Prazos (Item 5 - Obrigatório Jurídico)
        with st.container(border=True):
            st.markdown("**⏰ Próximos Prazos**")
            st.caption("5 prazos monitorados próximos")
            st.markdown(
                """
                | Processo | Prazo | Situação |
                | :--- | :--- | :--- |
                | **#2026-0145** | 2 dias | 🔴 Urgente |
                | **#2026-0182** | 5 dias | 🟡 Atenção |
                | **#2026-0191** | 12 dias | 🟢 Normal |
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. Gráfico de Linha Executivo de Atividade da IA (Largura Total)
    with st.container(border=True):
        st.markdown("**📈 Atividade da IA — Últimos 7 Dias**")
        st.caption("Volume de interações e consultas processadas pela inteligência artificial")
        
        # Gráfico executivo estilizado limpo em código estruturado profissional
        chart_data = """
        Requisições IA
         100 ┤                         ╭── (90)
          80 ┤                  ╭──────╯ (82)
          60 ┤          ╭───────╯ (68)
          40 ┤────╮─────╯ (55)
          20 ┤    ╰──── (45)
             └───────────────────────────────
                Seg  Ter  Qua  Qui  Sex
        """
        st.code(chart_data, language="text")

    st.markdown("<br>", unsafe_allow_html=True)
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
# ASSISTENTE IA (COM RESPOSTAS DINÂMICAS POR AGENTE)
# ============================================================

elif page == "Assistente IA":
    st.markdown(
        """
        <div class="hero">
            <h1>⚖️ Assistente IA — Inteligência Jurídica v3.1</h1>
            <p>
                Consulte a base jurídica, valide riscos e realize análises automatizadas com suporte de múltiplos agentes de IA.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Painel de Configuração da Análise
    with st.container(border=True):
        col_st1, col_st2 = st.columns([3, 1])
        with col_st1:
            st.markdown("⚙️ **Painel de Configuração da Análise**")
        with col_st2:
            st.markdown("<div style='text-align: right;'><span class='badge-green'>🟢 IA Conectada</span> <span style='font-size:0.75rem; color:#6c7890;'>(Gemini 1.5 Pro)</span></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        with cfg_c1:
            selected_agent = st.selectbox(
                "Agente:",
                [
                    "⚖️ Agente Jurídico",
                    "⚠️ Agente de Risco",
                    "📝 Agente de Resumo",
                    "💬 Agente Geral",
                    "🔎 RAG / Base Jurídica"
                ],
                key="sel_agent"
            )
            selected_mode = st.selectbox(
                "Modo de análise:",
                [
                    "Análise jurídica completa",
                    "Verificação de conformidade",
                    "Auditoria de cláusulas",
                    "Busca jurisprudencial"
                ],
                key="sel_mode"
            )
        with cfg_c2:
            selected_case = st.selectbox(
                "Caso:",
                [
                    "Processo #2026-0145",
                    "Processo #2026-0182",
                    "Processo #2026-0191",
                    "Nenhum / Geral"
                ],
                key="sel_case"
            )
            selected_depth = st.selectbox(
                "Nível de profundidade:",
                ["Detalhado", "Resumido", "Executivo", "Avançado (RAG estendido)"],
                key="sel_depth"
            )
        with cfg_c3:
            selected_doc = st.selectbox(
                "Documento:",
                [
                    "Contrato Cliente A.pdf",
                    "Petição Inicial.pdf",
                    "Contestação.docx",
                    "Todos os documentos do caso"
                ],
                key="sel_doc"
            )
            selected_source = st.selectbox(
                "Fonte de conhecimento:",
                [
                    "Caso específico",
                    "Toda a base jurídica + RAG",
                    "Documento específico",
                    "Busca RAG avançada"
                ],
                key="sel_source"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Executar análise jurídica", type="primary", use_container_width=True):
            st.session_state.pending_question = f"Realizar {selected_mode.lower()} utilizando o {selected_agent} focado no {selected_case} ({selected_doc})."
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Ações Rápidas Organizadas por Categorias
    st.markdown("### ⚡ Ações Rápidas")
    
    tab_cat1, tab_cat2, tab_cat3 = st.tabs(["📄 Documentos", "⚠️ Análise Jurídica", "⚖️ Produção Jurídica"])
    
    with tab_cat1:
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            if st.button("Analisar documento", use_container_width=True, key="q_doc1"):
                st.session_state.pending_question = "Faça uma análise detalhada e crítica do documento selecionado."
                st.rerun()
        with qc2:
            if st.button("Resumir documento", use_container_width=True, key="q_doc2"):
                st.session_state.pending_question = "Gere um resumo executivo completo do documento."
                st.rerun()
        with qc3:
            if st.button("Extrair cláusulas", use_container_width=True, key="q_doc3"):
                st.session_state.pending_question = "Extraia e categorize as principais cláusulas deste documento."
                st.rerun()
        with qc4:
            if st.button("Fazer perguntas", use_container_width=True, key="q_doc4"):
                st.session_state.pending_question = "Com base no documento, quais são as obrigações principais das partes?"
                st.rerun()

    with tab_cat2:
        qc5, qc6, qc7, qc8 = st.columns(4)
        with qc5:
            if st.button("Identificar riscos", use_container_width=True, key="q_an1"):
                st.session_state.pending_question = "Identifique todos os riscos contratuais, legais e processuais."
                st.rerun()
        with qc6:
            if st.button("Identificar obrigações", use_container_width=True, key="q_an2"):
                st.session_state.pending_question = "Liste de forma clara todas as obrigações e prazos de cada parte."
                st.rerun()
        with qc7:
            if st.button("Identificar prazos", use_container_width=True, key="q_an3"):
                st.session_state.pending_question = "Identifique todos os prazos processuais e contratuais críticos."
                st.rerun()
        with qc8:
            if st.button("Detectar inconsistências", use_container_width=True, key="q_an4"):
                st.session_state.pending_question = "Analise o texto buscando contradições ou inconsistências jurídicas."
                st.rerun()

    with tab_cat3:
        qc9, qc10, qc11, qc12 = st.columns(4)
        with qc9:
            if st.button("Gerar parecer preliminar", use_container_width=True, key="q_pr1"):
                st.session_state.pending_question = "Elabore um parecer jurídico preliminar fundamentado nas evidências."
                st.rerun()
        with qc10:
            if st.button("Gerar relatório", use_container_width=True, key="q_pr2"):
                st.session_state.pending_question = "Gere um relatório executivo estruturado com os pontos levantados."
                st.rerun()
        with qc11:
            if st.button("Gerar minuta", use_container_width=True, key="q_pr3"):
                st.session_state.pending_question = "Elabore uma minuta com base nos parâmetros do caso."
                st.rerun()
        with qc12:
            if st.button("Gerar síntese do caso", use_container_width=True, key="q_pr4"):
                st.session_state.pending_question = "Gere uma síntese objetiva para alinhamento com a equipe."
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Botão de limpar histórico
    col_cc1, _ = st.columns([1, 6])
    with col_cc1:
        if st.button("🗑️ Limpar Conversa", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if st.session_state.messages:
        st.markdown("### 💬 Histórico da Análise")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    pending = st.session_state.pop("pending_question", None)
    q = st.chat_input("Digite sua pergunta jurídica ou solicite uma análise...")
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
                with st.spinner("Executando pipeline: RAG → Retriever → Reranker → Agente IA..."):
                    # CORREÇÃO (item 2.5): o dropdown "Agente:" do painel de
                    # configuração era puramente decorativo — o orquestrador
                    # sempre rodava em mode="auto" (decisão só por palavra-chave
                    # na pergunta). Agora o agente escolhido pelo usuário é
                    # repassado de verdade para orchestrate(mode=...).
                    agent_mode_map = {
                        "Risco": "risk",
                        "Resumo": "summary",
                        "Geral": "general",
                        "Jurídico": "legal",
                        # "RAG / Base Jurídica" não tem um modo dedicado no
                        # orquestrador (é sobre a fonte de conhecimento, não
                        # sobre qual agente/prompt roda) — cai em detecção
                        # automática por conteúdo da pergunta.
                        "RAG": "auto",
                    }

                    selected_agent_label = st.session_state.get(
                        "sel_agent", "⚖️ Agente Jurídico"
                    )

                    selected_mode = "auto"
                    for keyword, mode_value in agent_mode_map.items():
                        if keyword in selected_agent_label:
                            selected_mode = mode_value
                            break

                    try:
                        result = call_orchestrator(
                            query=q,
                            org_id=user.get("organization_id"),
                            mode=selected_mode,
                            top_k=8,
                            rerank_k=5,
                        )
                    except Exception as e:
                        result = {"answer": f"Erro ao executar o orquestrador: {e}"}

                raw_response = str(result.get("answer", "") or "").strip()

                # CORREÇÃO (item 2.1): antes, qualquer resposta vazia OU em
                # modo demonstração era substituída por um parecer jurídico
                # fabricado na hora ("Modo Simulação Inteligente"), com
                # citações, riscos e recomendações inventados — indistinguível
                # de uma análise real aos olhos do usuário. Isso valia tanto
                # para "sem provedor configurado" quanto para falhas reais de
                # API (timeout, chave inválida, rate limit), que retornam
                # answer="" da mesma forma. Um sistema jurídico não pode
                # mascarar essas duas situações com texto que parece análise.
                #
                # Agora: mostramos exatamente o que aconteceu, sem inventar
                # conteúdo jurídico nenhum.

                is_demo_mode = (
                    "nenhum provedor llm está configurado" in raw_response.lower()
                    or "modo demonstração" in raw_response.lower()
                )

                if is_demo_mode:
                    st.info(
                        "🔧 **Nenhum provedor de IA está configurado neste ambiente.** "
                        "Esta é uma mensagem informativa do sistema, não uma análise jurídica.\n\n"
                        "Configure `LLM_PROVIDER=gemini` (com `GEMINI_API_KEY`) ou "
                        "`LLM_PROVIDER=openai` (com `OPENAI_API_KEY`) nas variáveis de "
                        "ambiente para obter respostas reais."
                    )
                    response = (
                        "_Nenhuma análise foi gerada: o provedor de IA não está "
                        "configurado neste ambiente. Veja o aviso acima._"
                    )
                elif not raw_response:
                    orchestrator_error = str(result.get("error") or "").strip()
                    st.error(
                        "❌ **Não foi possível gerar uma resposta.**\n\n"
                        + (
                            f"Detalhe técnico: {orchestrator_error}"
                            if orchestrator_error
                            else "O serviço de IA não retornou conteúdo. Tente novamente "
                                 "em instantes ou verifique os logs do sistema."
                        )
                    )
                    response = (
                        "_Nenhuma análise foi gerada devido a uma falha no serviço de IA. "
                        "Veja o erro acima._"
                    )
                else:
                    response = raw_response

                # Workspace de Resultado com Abas Internas Organizadas
                st.markdown("### 🤖 Resultado da Análise")
                res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📋 Resumo", "⚠️ Riscos", "📌 Evidências", "📚 Citações"])

                # CORREÇÃO (item 2.1 + 2.3): as quatro abas abaixo mostravam
                # dado 100% fabricado (risco fixo "Médio", confiança fixa
                # "94.5%", citações e evidências de exemplo hardcoded, mesmo
                # quando a resposta vinha de erro/modo demo). Agora usam
                # result["guard"], result["evaluation"] e result["citations"]
                # de verdade, retornados pelo orchestrator corrigido.

                guard = result.get("guard") or {}
                evaluation = result.get("evaluation") or {}
                real_citations = result.get("citations") or []

                overall = evaluation.get("overall")
                confidence_text = f"{overall * 100:.1f}%" if isinstance(overall, (int, float)) else "N/D"
                guard_approved = bool(guard.get("approved", False))
                guard_issues = guard.get("issues", []) or []

                with res_tab1:
                    st.markdown("#### Resumo Executivo")
                    st.markdown(response)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if guard_approved:
                        status_line = "🟢 Guard Agent: <b>Aprovada</b>"
                    elif guard_issues:
                        status_line = "🟡 Guard Agent: <b>Aprovada com ressalvas</b>"
                    else:
                        status_line = "⚪ Guard Agent: <b>Não avaliada</b>"

                    st.markdown(
                        f"""
                        <div style="background:#f0f4f8; padding:10px 14px; border-radius:8px; font-size:0.85rem;">
                            {status_line} &nbsp;&nbsp;|&nbsp;&nbsp;
                            🎯 Score de qualidade (real): <b>{confidence_text}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if guard_issues:
                        st.markdown("<br>", unsafe_allow_html=True)
                        for issue in guard_issues:
                            st.warning(issue)

                with res_tab2:
                    st.markdown("#### Riscos Identificados")
                    risk_result = result.get("risk") or {}
                    risk_answer = risk_result.get("answer")
                    if risk_answer:
                        st.markdown(risk_answer)
                    else:
                        st.info(
                            "Nenhuma análise de risco dedicada foi executada para esta "
                            "pergunta. Selecione o Agente de Risco ou peça explicitamente "
                            "uma análise de riscos para acionar esse agente."
                        )

                with res_tab3:
                    st.markdown("#### Evidências Recuperadas")
                    if not real_citations:
                        st.info("Nenhuma evidência foi recuperada da base para esta resposta.")
                    else:
                        for citation in real_citations:
                            if not isinstance(citation, dict):
                                continue
                            doc = citation.get("document", "Documento")
                            page = citation.get("page", "N/D")
                            content = citation.get("content", "")
                            score = citation.get("reranker_score")
                            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "N/D"
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #e0e6ed; padding: 12px; border-radius: 8px; background: #fafbfc; margin-bottom: 8px;">
                                    <b>[{citation.get('id', '?')}] {doc}</b> &middot; Página: {page} &middot; Score rerank: <b>{score_text}</b>
                                    <br><br>
                                    <blockquote style="margin: 0; color: #555; font-style: italic; border-left: 3px solid #1769e0; padding-left: 8px;">
                                        {content}
                                    </blockquote>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                with res_tab4:
                    render_citations(real_citations)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })

# ============================================================
# DOCUMENTOS & GESTÃO DA BASE DE CONHECIMENTO (VERSÃO CORRIGIDA)
# ============================================================

elif page == "Documentos":

    # CORREÇÃO (item 2.9): toda esta página era decorativa — o upload
    # nunca chamava ingest_document(), a biblioteca mostrava dois
    # documentos fixos ("Petição_Inicial.txt", "Contrato_..."), e os
    # KPIs eram números fixos. Agora tudo vem de services.documents
    # (dado real do SQLite) e o upload chama services.ingestion de
    # verdade (extração → chunking → embeddings → FAISS).

    org_id = user.get("organization_id")
    documents = list_documents(org_id) if org_id else []
    doc_status = document_status(org_id) if org_id else {"documents": 0, "chunks": 0}

    total_docs = doc_status.get("documents", len(documents))
    total_chunks = doc_status.get("chunks", 0)
    total_pages = sum(int(d.get("pages") or 0) for d in documents)
    ready_docs = sum(1 for d in documents if str(d.get("status", "")).lower() == "indexado")

    # 1️⃣ Cabeçalho e Indicador Superior do RAG
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("📄 Documentos")
        st.caption("Centralize contratos, petições, procurações e demais documentos jurídicos do seu escritório.")
    with head_col2:
        status_ok = total_docs > 0
        badge_bg = "#f0fdf4" if status_ok else "#fffbeb"
        badge_border = "#bbf7d0" if status_ok else "#fde68a"
        badge_color = "#15803d" if status_ok else "#92400e"
        badge_text = "🟢 Base jurídica operacional" if status_ok else "🟡 Nenhum documento indexado ainda"
        st.markdown(
            f"""
            <div style="background: {badge_bg}; border: 1px solid {badge_border}; padding: 10px; border-radius: 8px; text-align: right;">
                <span style="font-size: 0.8rem; color: {badge_color}; font-weight: 600;">{badge_text}</span><br>
                <span style="font-size: 0.75rem; color: #4b5563;">{total_docs} documento(s) • {total_chunks} chunks indexados</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2️⃣ KPIs da Base de Documentos (dado real)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        with st.container(border=True):
            st.markdown(f"📄 **{total_docs}**")
            st.caption("Documentos")
    with kpi2:
        with st.container(border=True):
            st.markdown(f"📚 **{total_pages}**")
            st.caption("Páginas Totais")
    with kpi3:
        with st.container(border=True):
            st.markdown(f"🧩 **{total_chunks}**")
            st.caption("Chunks Indexados")
    with kpi4:
        with st.container(border=True):
            st.markdown(f"🟢 **{ready_docs}**")
            st.caption("Prontos para RAG")

    st.markdown("<br>", unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # 3️⃣ Área de Upload — agora chama ingest_document() de verdade
        with st.container(border=True):
            st.markdown("📤 **Adicionar novos documentos**")
            st.caption("Arraste e solte seus arquivos abaixo ou clique para selecionar.")

            uploaded_file = st.file_uploader(
                "Carregar arquivos (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )

            use_ocr = st.checkbox(
                "☑ Usar OCR quando necessário (para documentos digitalizados / escaneados)",
                value=True,
            )

            if uploaded_file:
                st.markdown("<br>", unsafe_allow_html=True)

                # Evita reprocessar o mesmo arquivo a cada rerun do Streamlit
                # (o widget mantém o arquivo "presente" até ser removido).
                already_processed_key = f"doc_processed::{uploaded_file.name}::{uploaded_file.size}"

                if st.session_state.get(already_processed_key):
                    st.info(f"`{uploaded_file.name}` já foi processado nesta sessão.")
                elif st.button("📤 Processar e indexar documento", type="primary"):
                    with st.status("Processando documento na pipeline de IA...", expanded=True) as status:
                        try:
                            st.write("⏳ Extraindo texto, dividindo em chunks e gerando embeddings...")
                            result = ingest_document(
                                uploaded_file,
                                org_id,
                                use_ocr=use_ocr,
                            )
                            st.write(f"✓ {result.get('chunks', 0)} chunk(s) gerados a partir de {result.get('pages', 0)} página(s)")
                            if result.get("ocr_pages"):
                                st.write(f"✓ OCR aplicado em {result['ocr_pages']} página(s)")
                            st.write(f"✓ {result.get('indexed_chunks', 0)} chunk(s) indexados no FAISS")
                            status.update(
                                label="🟢 Documento pronto para consulta via RAG!",
                                state="complete",
                                expanded=False,
                            )
                            st.session_state[already_processed_key] = True
                            audit(
                                action="document_upload",
                                details={"filename": uploaded_file.name, "result": result},
                                organization_id=org_id,
                            )
                            st.rerun()
                        except ValueError as exc:
                            status.update(label="🟡 Não foi possível processar", state="error", expanded=True)
                            st.warning(str(exc))
                        except Exception as exc:
                            status.update(label="🔴 Falha na indexação", state="error", expanded=True)
                            st.error(f"Erro ao processar o documento: {exc}")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4️⃣ Biblioteca de Documentos — agora lista dado real
        st.markdown("### 📚 Biblioteca de documentos")

        filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
        with filter_col1:
            search_doc = st.text_input("Buscar documento...", placeholder="Digite o nome do arquivo...", label_visibility="collapsed")
        with filter_col2:
            type_filter = st.selectbox("Tipo", ["Todos os tipos", "PDF", "TXT", "DOCX"], label_visibility="collapsed")
        with filter_col3:
            status_filter = st.selectbox("Status", ["Todos", "Indexado", "Processando", "Erro na indexação"], label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)

        visible_documents = documents

        if search_doc:
            visible_documents = [
                d for d in visible_documents
                if search_doc.lower() in str(d.get("name", "")).lower()
            ]

        if type_filter != "Todos os tipos":
            visible_documents = [
                d for d in visible_documents
                if str(d.get("type", "")).upper() == type_filter.upper()
            ]

        if status_filter != "Todos":
            visible_documents = [
                d for d in visible_documents
                if str(d.get("status", "")) == status_filter
            ]

        if not visible_documents:
            st.info(
                "Nenhum documento encontrado. Envie um arquivo acima para começar."
                if not documents
                else "Nenhum documento corresponde aos filtros selecionados."
            )

        for doc in visible_documents:
            doc_id = doc.get("id")
            doc_name = doc.get("name", "Documento")
            doc_type = str(doc.get("type", "")).upper()
            doc_pages = doc.get("pages", 0)
            doc_chunks = doc.get("chunks", 0)
            doc_status_label = doc.get("status", "Desconhecido")

            icon = "📕" if doc_type == "PDF" else ("📘" if doc_type == "DOCX" else "📄")

            if doc_status_label == "Indexado":
                badge_class, badge_icon = "badge-green", "🟢"
            elif doc_status_label == "Processando":
                badge_class, badge_icon = "badge-orange", "🟡"
            else:
                badge_class, badge_icon = "badge-red", "🔴"

            with st.container(border=True):
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.markdown(f"{icon} **{doc_name}**")
                    st.caption(f"{doc_type} • {doc_pages} página(s) • {doc_chunks} chunk(s)")
                    st.markdown(f"<span class='{badge_class}'>{badge_icon} {doc_status_label}</span>", unsafe_allow_html=True)
                with d_col2:
                    if st.button("Analisar com IA", key=f"analyze_doc_{doc_id}", type="primary", use_container_width=True):
                        st.session_state.page = "Assistente IA"
                        st.session_state.pending_question = f"Faça uma análise detalhada de {doc_name}."
                        st.rerun()
                    if st.button("🗑️ Excluir", key=f"delete_doc_{doc_id}", use_container_width=True):
                        if delete_document(doc_id, org_id):
                            audit(
                                action="document_delete",
                                details={"document_id": doc_id, "filename": doc_name},
                                organization_id=org_id,
                                entity_type="document",
                                entity_id=doc_id,
                            )
                            st.rerun()

    with col_side:
        # 5️⃣ Base de Conhecimento da IA — dado real
        with st.container(border=True):
            st.markdown("🧠 **Base de Conhecimento**")
            st.caption(f"{total_docs} documento(s) disponível(is) para o motor RAG.")

            st.markdown("---")

            last_update = "N/D"
            if documents:
                last_update = documents[0].get("created_at", "N/D")

            st.markdown("**Métricas do Vector Store**")
            st.markdown(f"Chunks indexados: `{total_chunks}`")
            st.markdown(f"Documentos processados: `{total_docs}`")
            st.markdown(f"Última atualização: `{last_update}`")
            st.markdown(f"Status do Motor: `{'🟢 Operacional' if total_chunks > 0 else '🟡 Aguardando documentos'}`")

        # Painel Informativo Lateral: Conexão Documento → Assistente IA
        with st.container(border=True):
            st.markdown("⚡ **Ações Rápidas por Documento**")
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #4b5563; line-height: 1.6;">
                Ao clicar em <b>Analisar com IA</b> em qualquer documento da biblioteca, o sistema redireciona instantaneamente para o Assistente configurando:
                <br><br>
                • Resumo executivo automático<br>
                • Identificação de riscos contratuais<br>
                • Extração de cláusulas e obrigações<br>
                • Monitoramento de prazos críticos
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
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
# GESTÃO DE PROCESSOS (CENTRO DE CONTEXTO DO SISTEMA)
# ============================================================

elif page == "Processos":

    # CORREÇÃO (item 2.4): esta página inteira era decorativa — os três
    # processos (#2026-0145, #2026-0182, #2026-0191), riscos, prazos e
    # histórico eram HTML estático, mesmo com services.cases pronto e
    # importado (create_case/list_cases nunca eram chamados). Agora usa
    # dado real do banco. A tabela `cases` (db.py) só guarda title,
    # client, category, priority, status e created_at — não guarda
    # risco, prazo, documentos vinculados nem histórico de análises.
    # Por isso essas seções foram removidas em vez de mantidas com
    # dado fabricado (mesmo princípio do item 2.1): melhor não mostrar
    # a informação do que mostrar uma inventada.

    org_id = user.get("organization_id")

    selected_process_id = st.session_state.get("active_process_id", None)

    if selected_process_id:
        # ========================================================
        # VISÃO DETALHADA DO PROCESSO SELECIONADO (dado real)
        # ========================================================

        case = get_case(org_id, selected_process_id) if org_id else None

        col_back, col_actions = st.columns([4, 1])
        with col_back:
            if st.button("← Voltar para a lista de processos"):
                st.session_state.pop("active_process_id", None)
                st.rerun()
        with col_actions:
            if case and st.button("⚡ Analisar com IA", type="primary", use_container_width=True):
                st.session_state.page = "Assistente IA"
                st.session_state.pending_question = (
                    f"Faça uma análise sobre o processo '{case.get('title')}' "
                    f"do cliente {case.get('client')}."
                )
                st.rerun()

        if not case:
            st.warning("Processo não encontrado.")
        else:
            st.markdown(f"## ⚖️ {case.get('title', 'Processo')}")
            st.caption(
                f"Cliente: **{case.get('client', 'N/D')}** &middot; "
                f"Categoria: **{case.get('category', 'N/D')}** &middot; "
                f"Status: **{case.get('status', 'N/D')}**"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            pk1, pk2, pk3 = st.columns(3)
            priority = case.get("priority", "N/D")
            priority_badge = {
                "Crítica": "🔴", "Alta": "🟠", "Média": "🟡", "Baixa": "🟢",
            }.get(priority, "⚪")
            with pk1:
                with st.container(border=True):
                    st.markdown(f"{priority_badge} **Prioridade: {priority}**")
            with pk2:
                with st.container(border=True):
                    st.markdown(f"📌 **Status: {case.get('status', 'N/D')}**")
            with pk3:
                with st.container(border=True):
                    st.markdown(f"📅 **Aberto em: {case.get('created_at', 'N/D')}**")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### Atualizar status")
            new_status = st.selectbox(
                "Status do processo",
                CASE_STATUSES,
                index=CASE_STATUSES.index(case.get("status")) if case.get("status") in CASE_STATUSES else 0,
                label_visibility="collapsed",
            )
            if st.button("Salvar status"):
                update_result = update_case_status(org_id, selected_process_id, new_status)
                if update_result.get("updated"):
                    audit(
                        action="case_status_update",
                        details={"case_id": selected_process_id, "new_status": new_status},
                        organization_id=org_id,
                        entity_type="case",
                        entity_id=selected_process_id,
                    )
                    st.success("Status atualizado.")
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "Vínculo com documentos, riscos e prazos por processo ainda não "
                "está implementado no schema atual — só os campos acima existem "
                "de fato no banco. Use o botão 'Analisar com IA' para consultar "
                "a base de documentos com contexto deste processo."
            )

    else:
        # ========================================================
        # LISTAGEM GERAL DE PROCESSOS (dado real)
        # ========================================================

        st.title("⚖️ Gestão de Processos")
        st.caption("Central de casos jurídicos do escritório.")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("➕ Novo processo"):
            with st.form("new_case_form", clear_on_submit=True):
                nc_title = st.text_input("Título do processo")
                nc_client = st.text_input("Cliente")
                nc_category = st.text_input("Categoria (ex.: Trabalhista, Societário, Contratual)")
                nc_priority = st.selectbox("Prioridade", CASE_PRIORITIES)
                nc_submitted = st.form_submit_button("Criar processo", type="primary")

                if nc_submitted:
                    try:
                        created = create_case(org_id, nc_title, nc_client, nc_category, nc_priority)
                        audit(
                            action="case_create",
                            details={"case_id": created.get("case_id"), "title": nc_title},
                            organization_id=org_id,
                            entity_type="case",
                            entity_id=created.get("case_id"),
                        )
                        st.success(f"Processo '{nc_title}' criado.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        f_col1, f_col2 = st.columns([3, 2])
        with f_col1:
            search_query = st.text_input(
                "Buscar processo...",
                placeholder="Digite o título, cliente ou categoria...",
                label_visibility="collapsed",
            )
        with f_col2:
            status_filter = st.selectbox(
                "Filtro de status",
                ["Todos"] + CASE_STATUSES,
                label_visibility="collapsed",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if org_id:
            cases = search_cases(org_id, search_query) if search_query else list_cases(org_id)
        else:
            cases = []

        if status_filter != "Todos":
            cases = [c for c in cases if c.get("status") == status_filter]

        if not cases:
            st.info(
                "Nenhum processo cadastrado ainda. Use '➕ Novo processo' acima para começar."
                if not search_query
                else "Nenhum processo encontrado para essa busca."
            )

        priority_badges = {
            "Crítica": ("badge-red", "🔴"),
            "Alta": ("badge-orange", "🟠"),
            "Média": ("badge-orange", "🟡"),
            "Baixa": ("badge-green", "🟢"),
        }

        for case in cases:
            case_id = case.get("id")
            badge_class, badge_icon = priority_badges.get(case.get("priority"), ("badge-green", "⚪"))

            with st.container(border=True):
                pc_col1, pc_col2 = st.columns([3, 1])
                with pc_col1:
                    st.markdown(f"### {case.get('title', 'Processo')}")
                    st.markdown(
                        f"Cliente: **{case.get('client', 'N/D')}** &middot; "
                        f"Categoria: **{case.get('category', 'N/D')}**"
                    )
                    st.markdown(
                        f"Status: **{case.get('status', 'N/D')}** &middot; "
                        f"Prioridade: <span class='{badge_class}'>{badge_icon} {case.get('priority', 'N/D')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span style='font-size: 0.8rem; color: #64748b;'>Aberto em {case.get('created_at', 'N/D')}</span>",
                        unsafe_allow_html=True,
                    )
                with pc_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Ver processo", key=f"btn_open_{case_id}", use_container_width=True):
                        st.session_state.active_process_id = case_id
                        st.rerun()
                    if st.button("Analisar com IA", key=f"btn_ai_{case_id}", type="primary", use_container_width=True):
                        st.session_state.page = "Assistente IA"
                        st.session_state.pending_question = (
                            f"Faça uma análise sobre o processo '{case.get('title')}' "
                            f"do cliente {case.get('client')}."
                        )
                        st.rerun()
