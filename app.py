
"""
Assistente Jurídico IA SaaS V3.1
Interface premium — Dark Legal Tech

Este arquivo é uma versão visualmente redesenhada do app principal.
Mantém a integração com os serviços existentes do projeto:
- autenticação
- banco de dados
- documentos / ingestão
- RAG
- reranker
- orquestrador de agentes
- processos
- auditoria

Substitua o app.py somente depois de testar esta versão localmente.
"""

from __future__ import annotations

import inspect
import io
import traceback
from datetime import datetime, date, timedelta
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import init_db, seed_demo, get_connection
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
from services.rag_pipeline import rag_answer, retrieve_and_rerank
from services.ai_orchestrator import orchestrate, risk_analysis
from services.ai import ai_status, clear_ai_cache
from services.auth import authenticate, get_current_user, logout
from services.ingestion import ingest_document


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Assistente Jurídico IA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
<style>
:root{
    --bg:#020b1d;
    --bg2:#061630;
    --panel:#071d42;
    --panel2:#082653;
    --line:#173d72;
    --blue:#1685ff;
    --blue2:#2563eb;
    --cyan:#22d3ee;
    --purple:#7c3aed;
    --pink:#ec4899;
    --teal:#14b8a6;
    --green:#10d6a0;
    --orange:#f59e0b;
    --red:#f43f5e;
    --text:#f5f9ff;
    --muted:#91a8c8;
}

/* ---------- APP ---------- */
.stApp{
    background:
        radial-gradient(circle at 80% 10%, rgba(37,99,235,.12), transparent 25%),
        radial-gradient(circle at 30% 90%, rgba(124,58,237,.08), transparent 30%),
        linear-gradient(135deg,#020a19 0%,#041632 55%,#031024 100%);
    color:var(--text);
}

.block-container{
    max-width:1500px;
    padding-top:1.0rem;
    padding-bottom:2.5rem;
}

/* ---------- HEADER ---------- */
[data-testid="stHeader"]{
    background:rgba(2,11,29,.72);
    border-bottom:1px solid rgba(42,92,153,.35);
}

[data-testid="stToolbar"]{
    background:transparent;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"]{
    background:
        radial-gradient(circle at 20% 0%,rgba(37,99,235,.18),transparent 28%),
        linear-gradient(180deg,#031127 0%,#041a3c 58%,#03132e 100%);
    border-right:1px solid #153d70;
}

[data-testid="stSidebar"] > div:first-child{
    padding-top:1rem;
}

/* ---------- SIDEBAR PREMIUM V3.2 ---------- */
[data-testid="stSidebar"]{
    min-width:290px !important;
    max-width:290px !important;
}
[data-testid="stSidebar"] > div:first-child{
    padding:1rem .85rem 1.5rem !important;
}
[data-testid="stSidebar"] .stButton{ margin:5px 0 !important; }
[data-testid="stSidebar"] .stButton > button{
    min-height:44px !important;
    border-radius:12px !important;
    border:1px solid rgba(61,132,224,.28) !important;
    background:linear-gradient(180deg,rgba(10,39,82,.92),rgba(5,25,57,.92)) !important;
    color:#eef6ff !important;
    font-weight:650 !important;
    text-align:left !important;
    padding:8px 13px !important;
    box-shadow:0 6px 18px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.045) !important;
    transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover{
    transform:translateX(3px) !important;
    border-color:rgba(48,145,255,.75) !important;
    box-shadow:0 9px 24px rgba(0,102,255,.24), inset 0 1px 0 rgba(255,255,255,.06) !important;
}
[data-testid="stSidebar"] .sidebar-active .stButton > button{
    background:linear-gradient(90deg,#075fe6,#124cc4) !important;
    border-color:#2188ff !important;
    box-shadow:0 9px 26px rgba(0,100,255,.34) !important;
}
.sidebar-section-space{
    margin-top:16px;
    padding-top:10px;
    border-top:1px solid rgba(62,111,173,.24);
}
.top-search-wrap [data-testid="stTextInput"] input{
    height:42px !important;
    border-radius:13px !important;
    border:1px solid #1d5595 !important;
    background:linear-gradient(180deg,rgba(7,32,69,.96),rgba(3,19,44,.96)) !important;
    color:#f4f8ff !important;
    box-shadow:0 7px 24px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.04) !important;
}
.top-search-wrap [data-testid="stTextInput"] input:focus{
    border-color:#318cff !important;
    box-shadow:0 0 0 2px rgba(49,140,255,.16), 0 9px 28px rgba(0,80,220,.20) !important;
}
.search-results{
    margin:8px 0 16px;
    padding:14px;
    border:1px solid #1b4e8c;
    border-radius:14px;
    background:linear-gradient(180deg,rgba(7,29,63,.98),rgba(3,18,42,.98));
    box-shadow:0 14px 35px rgba(0,0,0,.22);
}
.search-result-item{
    padding:10px 12px;
    border-radius:10px;
    border:1px solid rgba(51,110,180,.20);
    background:rgba(11,40,79,.55);
    margin-top:7px;
}
.ai-status-card{
    border:1px solid rgba(41,126,218,.42);
    border-radius:14px;
    padding:13px 15px;
    background:linear-gradient(135deg,rgba(5,29,64,.95),rgba(8,39,82,.75));
    box-shadow:0 10px 28px rgba(0,0,0,.18);
}
.ai-status-ok{color:#2ee6ad;font-weight:800}
.ai-status-warn{color:#ffbd4a;font-weight:800}
.ai-status-error{color:#ff6680;font-weight:800}


[data-testid="stSidebar"] *{
    color:#edf6ff;
}

[data-testid="stSidebar"] .stRadio > label{
    display:none;
}

[data-testid="stSidebar"] [role="radiogroup"]{
    gap:4px;
}

[data-testid="stSidebar"] [role="radio"]{
    min-height:42px;
    padding:7px 11px;
    border-radius:11px;
    border:1px solid transparent;
    transition:all .18s ease;
}

[data-testid="stSidebar"] [role="radio"]:hover{
    background:rgba(37,99,235,.16);
    border-color:rgba(59,130,246,.20);
    transform:translateX(2px);
}

[data-testid="stSidebar"] [role="radio"][aria-checked="true"]{
    background:linear-gradient(90deg,#0758d9 0%,#124cc4 100%);
    border-color:#1678ff;
    box-shadow:0 7px 24px rgba(0,102,255,.25);
}

[data-testid="stSidebar"] [role="radio"] > div:first-child{
    display:none;
}

[data-testid="stSidebar"] [role="radio"] p{
    font-size:.88rem;
    font-weight:600;
}

/* ---------- SIDEBAR BRAND ---------- */
.legal-brand{
    padding:4px 4px 18px;
    border-bottom:1px solid rgba(62,111,173,.35);
    margin-bottom:16px;
}

.legal-brand-row{
    display:flex;
    align-items:center;
    gap:11px;
}

.legal-logo{
    width:45px;
    height:45px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:27px;
    border-radius:13px;
    background:linear-gradient(145deg,#f8c94d,#b77717);
    box-shadow:0 8px 24px rgba(245,158,11,.20);
}

.legal-title{
    font-size:1.08rem;
    font-weight:800;
    letter-spacing:-.02em;
}

.legal-sub{
    margin-top:2px;
    color:#7eb8ff;
    font-size:.68rem;
}

/* ---------- GROUP LABELS ---------- */
.nav-group{
    color:#4fa4ff;
    font-size:.66rem;
    font-weight:800;
    letter-spacing:.10em;
    margin:15px 6px 7px;
}

/* ---------- PROFILE ---------- */
.sidebar-profile{
    border-top:1px solid rgba(62,111,173,.35);
    margin-top:18px;
    padding-top:16px;
}

.profile-row{
    display:flex;
    align-items:center;
    gap:10px;
}

.avatar{
    width:38px;
    height:38px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(145deg,#1e5fd5,#102f70);
    border:1px solid #2c6bc4;
    font-size:19px;
}

.profile-name{font-size:.82rem;font-weight:750}
.profile-role{font-size:.67rem;color:#8ba8ca}
.online{
    margin-top:5px;
    color:#12dca5;
    font-size:.67rem;
    font-weight:700;
}

.version{
    color:#6683a8;
    text-align:right;
    font-size:.64rem;
    margin-top:12px;
}

/* ---------- TOP SEARCH ---------- */
.top-search{
    border:1px solid #24548f;
    background:rgba(7,29,66,.72);
    border-radius:14px;
    padding:10px 15px;
    color:#9fc0e7;
    font-size:.82rem;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.015);
}

/* ---------- TITLES ---------- */
.page-title{
    font-size:2rem;
    line-height:1.1;
    font-weight:850;
    letter-spacing:-.045em;
    margin:10px 0 4px;
}

.page-subtitle{
    color:#8facd0;
    font-size:.88rem;
    margin-bottom:20px;
}

/* ---------- CARDS ---------- */
.glass-card{
    background:linear-gradient(145deg,rgba(7,31,70,.96),rgba(4,22,51,.94));
    border:1px solid rgba(38,100,174,.58);
    border-radius:15px;
    padding:18px;
    box-shadow:
        0 14px 35px rgba(0,0,0,.18),
        inset 0 1px 0 rgba(255,255,255,.025);
}

.section-card{
    background:rgba(5,24,54,.88);
    border:1px solid #173e73;
    border-radius:15px;
    padding:17px;
    box-shadow:0 12px 28px rgba(0,0,0,.16);
}

.card-title{
    font-size:.95rem;
    font-weight:800;
    color:#f5f9ff;
}

.card-sub{
    color:#7594b9;
    font-size:.68rem;
    margin-top:2px;
}

/* ---------- KPI ---------- */
.kpi{
    min-height:143px;
    padding:17px;
    border-radius:15px;
    border:1px solid rgba(50,118,201,.60);
    position:relative;
    overflow:hidden;
}

.kpi:after{
    content:"";
    position:absolute;
    width:120px;
    height:70px;
    right:-25px;
    bottom:-28px;
    border-radius:50%;
    border:1px solid rgba(96,165,250,.22);
    transform:rotate(-16deg);
}

.kpi-blue{
    background:linear-gradient(145deg,#06377d,#07336c);
}
.kpi-purple{
    background:linear-gradient(145deg,#151080,#25146e);
}
.kpi-red{
    background:linear-gradient(145deg,#3d1539,#42152e);
}
.kpi-teal{
    background:linear-gradient(145deg,#034f5b,#075264);
}

.kpi-icon{
    width:36px;
    height:36px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:17px;
    background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.13);
}

.kpi-label{
    margin-top:10px;
    color:#b8cbea;
    font-size:.73rem;
    font-weight:700;
}

.kpi-value{
    font-size:1.85rem;
    line-height:1;
    margin-top:7px;
    font-weight:850;
}

.kpi-trend{
    color:#14e3ad;
    font-size:.72rem;
    margin-top:9px;
    font-weight:700;
}

.kpi-muted{
    color:#7594b9;
    font-size:.66rem;
}

/* ---------- BADGES ---------- */
.badge{
    display:inline-block;
    border-radius:999px;
    padding:4px 9px;
    font-size:.66rem;
    font-weight:800;
}
.badge-green{background:rgba(16,214,160,.14);color:#20e6b1;border:1px solid rgba(16,214,160,.25)}
.badge-orange{background:rgba(245,158,11,.14);color:#ffc14b;border:1px solid rgba(245,158,11,.25)}
.badge-red{background:rgba(244,63,94,.14);color:#ff718a;border:1px solid rgba(244,63,94,.25)}
.badge-blue{background:rgba(59,130,246,.14);color:#67a8ff;border:1px solid rgba(59,130,246,.25)}

/* ---------- ALERTS ---------- */
.alert-row{
    padding:11px 8px;
    border-bottom:1px solid rgba(48,92,147,.35);
}

.alert-row:last-child{border-bottom:0}

.alert-icon{
    display:inline-flex;
    width:29px;
    height:29px;
    align-items:center;
    justify-content:center;
    border-radius:9px;
    background:rgba(244,63,94,.18);
    margin-right:8px;
}

.alert-title{
    font-size:.78rem;
    font-weight:750;
}

.alert-desc{
    margin-left:38px;
    color:#7895b8;
    font-size:.66rem;
}

/* ---------- ACTIVITY ---------- */
.activity{
    padding:11px 0;
    border-bottom:1px solid rgba(48,92,147,.3);
}
.activity:last-child{border-bottom:0}

.activity-icon{
    width:31px;
    height:31px;
    border-radius:9px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:#103c7a;
    margin-right:9px;
}

/* ---------- AI PANEL ---------- */
.ai-panel{
    background:
        radial-gradient(circle at 100% 0%,rgba(37,99,235,.22),transparent 40%),
        linear-gradient(145deg,#062658,#061a3c);
    border:1px solid #245e9f;
    border-radius:16px;
    padding:18px;
    box-shadow:0 15px 35px rgba(0,0,0,.20);
}

.ai-online{
    float:right;
    color:#0ee5a8;
    background:rgba(16,214,160,.13);
    border:1px solid rgba(16,214,160,.25);
    border-radius:999px;
    padding:4px 10px;
    font-size:.65rem;
    font-weight:800;
}

.ai-input{
    border:1px solid #9b5cff;
    box-shadow:0 0 0 2px rgba(124,58,237,.08);
    background:#091f47;
    border-radius:12px;
    padding:12px;
    color:#8fa9c9;
    font-size:.76rem;
}

/* ---------- BUTTONS ---------- */
.stButton > button{
    border-radius:10px;
    border:1px solid #1d4c84;
    background:#082653;
    color:#eaf4ff;
    font-weight:650;
    min-height:39px;
}

.stButton > button:hover{
    border-color:#2e8cff;
    background:#0a3470;
    color:#fff;
    box-shadow:0 7px 20px rgba(0,104,255,.16);
}

.stButton > button[kind="primary"]{
    background:linear-gradient(90deg,#075ee2,#1769e0);
    border-color:#1678ff;
}

/* ---------- INPUTS ---------- */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input{
    background:#071f45 !important;
    color:#eef6ff !important;
    border-color:#204f88 !important;
}

[data-baseweb="select"] span{
    color:#eef6ff !important;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"]{
    gap:5px;
    background:transparent;
}
.stTabs [data-baseweb="tab"]{
    color:#87a4c7;
    border-radius:9px;
    padding:8px 13px;
}
.stTabs [aria-selected="true"]{
    color:#fff !important;
    background:#0b326b;
}

/* ---------- TABLES / DATAFRAMES ---------- */
[data-testid="stDataFrame"]{
    border:1px solid #1a477e;
    border-radius:12px;
    overflow:hidden;
}

/* ---------- CHAT ---------- */
[data-testid="stChatMessage"]{
    background:rgba(7,29,66,.82);
    border:1px solid #183e72;
    border-radius:14px;
}

/* ---------- FOOTER ---------- */
.footer-banner{
    margin-top:16px;
    border-radius:14px;
    padding:12px 18px;
    border:1px solid #1e5bb2;
    background:
        radial-gradient(circle at 90% 100%,rgba(124,58,237,.30),transparent 30%),
        linear-gradient(90deg,#0647a5,#08295b);
    color:#dceaff;
    font-size:.76rem;
}

/* ---------- MOBILE ---------- */
@media(max-width:900px){
    .page-title{font-size:1.55rem}
    .block-container{padding-left:1rem;padding-right:1rem}
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
    return {"answer": str(value or ""), "citations": [], "retrieved": [], "reranked": []}


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
        return {"answer": "Informe uma pergunta.", "citations": []}

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
        params = signature.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        if accepts_kwargs:
            filtered = {k: v for k, v in kwargs.items() if v is not None}
        else:
            filtered = {
                k: v for k, v in kwargs.items()
                if k in params and v is not None
            }

        if "query" in params:
            filtered.pop("question", None)
        elif "question" in params:
            filtered.pop("query", None)

        if "org_id" in params:
            filtered.pop("organization_id", None)
        elif "organization_id" in params:
            filtered.pop("org_id", None)

        result = safe_dict(orchestrate(**filtered))
        result.setdefault("answer", "")
        result.setdefault("citations", [])
        result.setdefault("retrieved", [])
        result.setdefault("reranked", [])
        result.setdefault("agent", "general")
        result.setdefault("intent", "general")
        result.setdefault("guard", {"approved": False, "issues": []})
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


def get_counts(org_id):
    try:
        with get_connection() as conn:
            docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE organization_id=?",
                (org_id,),
            ).fetchone()[0]
            cases = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE organization_id=?",
                (org_id,),
            ).fetchone()[0]
        return int(docs), int(cases)
    except Exception:
        return 0, 0


def nav_button(page_name: str, label: str, icon: str, key: str):
    """Fallback helper for future custom navigation."""
    return st.button(
        f"{icon}  {label}",
        key=key,
        use_container_width=True,
    )


def metric_card(icon, label, value, trend, subtitle, css):
    st.markdown(
        f"""
        <div class="kpi {css}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-trend">{trend}</div>
            <div class="kpi-muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon, title, subtitle=""):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:9px;margin-bottom:12px;">
            <div style="font-size:1.15rem;">{icon}</div>
            <div>
                <div class="card-title">{title}</div>
                {f'<div class="card-sub">{subtitle}</div>' if subtitle else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _to_date(value):
    if not value:
        return None
    try:
        if isinstance(value, date):
            return value
        return pd.to_datetime(str(value), errors="coerce").date()
    except Exception:
        return None


def load_report_data(org_id):
    """Carrega dados reais disponíveis no banco para o painel de relatórios."""
    try:
        cases = list_cases(org_id) or []
    except Exception:
        cases = []
    try:
        documents = list_documents(org_id) or []
    except Exception:
        documents = []

    # Prazos atuais da página operacional.
    deadlines = [
        {"date": "10/09/2026", "title": "Processo #2026-0145", "description": "Manifestação processual", "priority": "Alto"},
        {"date": "11/09/2026", "title": "Documento pendente", "description": "Assinatura / revisão", "priority": "Médio"},
        {"date": "12/09/2026", "title": "Análise contratual", "description": "Revisão jurídica", "priority": "Médio"},
    ]

    risks = []
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT action, details, created_at FROM audit_logs "
                "WHERE organization_id=? ORDER BY created_at DESC LIMIT 500",
                (org_id,),
            ).fetchall()
        for row in rows:
            action = str(row[0] or "").lower()
            if "risk" in action or "risco" in action:
                risks.append({"action": row[0], "details": row[1], "created_at": row[2]})
    except Exception:
        # O projeto atual pode não registrar análises de risco no audit_logs.
        risks = []

    return cases, documents, risks, deadlines


def report_cases_dataframe(cases):
    rows = []
    for c in cases:
        rows.append({
            "ID": c.get("id", ""),
            "Processo": c.get("title", "Processo"),
            "Cliente": c.get("client", ""),
            "Categoria": c.get("category", ""),
            "Prioridade": c.get("priority", ""),
            "Status": c.get("status", ""),
            "Criado em": c.get("created_at", c.get("createdAt", "")),
        })
    return pd.DataFrame(rows, columns=["ID", "Processo", "Cliente", "Categoria", "Prioridade", "Status", "Criado em"])


def build_report_csv(cases, documents, risks, deadlines):
    case_df = report_cases_dataframe(cases)
    doc_rows = [{
        "Documento": d.get("name", d.get("filename", "Documento")),
        "Status": d.get("status", ""),
        "Páginas": d.get("pages", 0),
        "Chunks": d.get("chunks", 0),
    } for d in documents]
    doc_df = pd.DataFrame(doc_rows, columns=["Documento", "Status", "Páginas", "Chunks"])
    deadline_df = pd.DataFrame(deadlines)

    out = io.StringIO()
    out.write("RELATÓRIO EXECUTIVO - ASSISTENTE JURÍDICO IA V3.1\n\n")
    out.write("PROCESSOS\n")
    case_df.to_csv(out, index=False, sep=";")
    out.write("\nDOCUMENTOS\n")
    doc_df.to_csv(out, index=False, sep=";")
    out.write("\nRISCOS REGISTRADOS\n")
    if risks:
        pd.DataFrame(risks).to_csv(out, index=False, sep=";")
    else:
        out.write("Nenhuma análise de risco registrada no banco.\n")
    out.write("\nPRAZOS MONITORADOS\n")
    deadline_df.to_csv(out, index=False, sep=";")
    return out.getvalue().encode("utf-8-sig")



def build_report_excel(summary, cases_df, documents, risks, deadlines):
    """Gera um Excel profissional com resumo e uma aba para cada conjunto de dados."""
    buffer = io.BytesIO()

    doc_rows = [{
        "Documento": d.get("name", d.get("filename", "Documento")),
        "Status": d.get("status", ""),
        "Páginas": d.get("pages", 0),
        "Chunks": d.get("chunks", 0),
    } for d in documents]
    doc_df = pd.DataFrame(doc_rows, columns=["Documento", "Status", "Páginas", "Chunks"])

    risk_df = pd.DataFrame(risks) if risks else pd.DataFrame(columns=["action", "details", "created_at"])
    deadline_df = pd.DataFrame(deadlines) if deadlines else pd.DataFrame(columns=["date", "title", "description", "priority"])

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df = pd.DataFrame([
            ["Processos", summary.get("processes", 0)],
            ["Documentos", summary.get("documents", 0)],
            ["Riscos registrados", summary.get("risks", 0)],
            ["Prazos", summary.get("deadlines", 0)],
            ["Alta prioridade", summary.get("high_priority", 0)],
            ["Gerado em", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ], columns=["Indicador", "Valor"])
        summary_df.to_excel(writer, sheet_name="Resumo", index=False)
        cases_df.to_excel(writer, sheet_name="Processos", index=False)
        doc_df.to_excel(writer, sheet_name="Documentos", index=False)
        risk_df.to_excel(writer, sheet_name="Riscos", index=False)
        deadline_df.to_excel(writer, sheet_name="Prazos", index=False)

        wb = writer.book
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        header_fill = PatternFill("solid", fgColor="0B3B78")
        header_font = Font(color="FFFFFF", bold=True)

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for col in range(1, ws.max_column + 1):
                max_len = 0
                for row in ws.iter_rows(min_col=col, max_col=col):
                    value = row[0].value
                    max_len = max(max_len, len(str(value or "")))
                ws.column_dimensions[get_column_letter(col)].width = min(max(max_len + 2, 12), 45)

    return buffer.getvalue()

def build_report_pdf(summary, cases_df, doc_count, risks_count, deadlines):
    """Gera PDF simples e profissional. Importação tardia para não quebrar o app."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#0B3B78"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [
        Paragraph("Assistente Juridico IA", styles["ReportTitle"]),
        Paragraph("Relatório executivo da operação jurídica · V3.1", styles["Normal"]),
        Spacer(1, 16),
    ]
    kpi_data = [
        ["Processos", str(summary["processes"]), "Documentos", str(doc_count)],
        ["Riscos registrados", str(risks_count), "Prazos", str(summary["deadlines"])],
    ]
    table = Table(kpi_data, colWidths=[110, 65, 110, 65])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EAF2FF")),
        ("BOX", (0,0), (-1,-1), .7, colors.HexColor("#A9C6EA")),
        ("INNERGRID", (0,0), (-1,-1), .4, colors.HexColor("#C9D9EE")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),
        ("ALIGN", (1,0), (1,-1), "CENTER"),
        ("ALIGN", (3,0), (3,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
    ]))
    story += [table, Spacer(1, 18), Paragraph("Processos", styles["Heading2"])]
    data = [["Processo", "Categoria", "Status", "Prioridade"]]
    for _, r in cases_df.head(30).iterrows():
        data.append([str(r.get("Processo", ""))[:45], str(r.get("Categoria", ""))[:18], str(r.get("Status", ""))[:18], str(r.get("Prioridade", ""))[:12]])
    if len(data) == 1:
        data.append(["Nenhum processo encontrado", "—", "—", "—"])
    t2 = Table(data, repeatRows=1, colWidths=[210, 100, 95, 75])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B3B78")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#B8CBE3")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F8FD")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [t2, Spacer(1, 18), Paragraph("Prazos monitorados", styles["Heading2"])]
    ddata = [["Data", "Item", "Descrição", "Prioridade"]] + [[x["date"], x["title"], x["description"], x["priority"]] for x in deadlines]
    t3 = Table(ddata, repeatRows=1, colWidths=[70, 150, 150, 70])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B3B78")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#B8CBE3")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F8FD")]),
    ]))
    story += [t3, Spacer(1, 16), Paragraph("Observação: os indicadores são calculados a partir dos dados disponíveis na organização. Riscos só são contabilizados quando registrados no banco de auditoria.", styles["Small"])]
    doc.build(story)
    return buffer.getvalue()


def report_status_chart(cases):
    df = report_cases_dataframe(cases)
    if df.empty or df["Status"].fillna("").eq("").all():
        labels, values = ["Sem dados"], [1]
    else:
        counts = df["Status"].fillna("Sem status").replace("", "Sem status").value_counts()
        labels, values = counts.index.tolist(), counts.values.tolist()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.68, textinfo="none", marker=dict(line=dict(color="#071a37", width=2)))])
    fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=True, legend=dict(font=dict(color="#b9cbe3", size=10)))
    return fig


def report_category_chart(cases):
    df = report_cases_dataframe(cases)
    if df.empty:
        labels, values = ["Sem dados"], [0]
    else:
        counts = df["Categoria"].fillna("Não informada").replace("", "Não informada").value_counts().head(8)
        labels, values = counts.index.tolist(), counts.values.tolist()
    fig = go.Figure(data=[go.Bar(x=values, y=labels, orientation="h")])
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#a9c0df", size=10), xaxis=dict(showgrid=True, gridcolor="rgba(47,88,140,.25)"), yaxis=dict(showgrid=False))
    return fig

def plot_dark_line():
    x = ["03/09", "04/09", "05/09", "06/09", "07/09", "08/09", "09/09"]
    y = [7, 8.5, 12.5, 11.7, 13.4, 16.2, 18]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#4f8cff", width=3, shape="spline"),
            marker=dict(color="#8ec5ff", size=6),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,.20)",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=235,
        margin=dict(l=8, r=8, t=10, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#91a8c8", size=10),
        xaxis=dict(
            showgrid=False,
            linecolor="#183e72",
            tickfont=dict(color="#7d9bc0"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(47,88,140,.25)",
            zeroline=False,
            tickfont=dict(color="#7d9bc0"),
        ),
        showlegend=False,
    )
    return fig


def plot_status_donut():
    labels = ["Em andamento", "Em análise", "Pendente", "Arquivado"]
    values = [12, 5, 4, 3]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=.68,
                textinfo="none",
                marker=dict(
                    colors=["#1785ff", "#7c3aed", "#f59e0b", "#647da2"],
                    line=dict(color="#071a37", width=2),
                ),
            )
        ]
    )
    fig.update_layout(
        height=205,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(
                text="<b>24</b><br><span style='font-size:10px'>Total</span>",
                x=.5,
                y=.5,
                showarrow=False,
                font=dict(color="#fff", size=21),
            )
        ],
    )
    return fig


# ============================================================
# DB INIT
# ============================================================

try:
    init_db()
    seed_demo()
except Exception:
    pass


# ============================================================
# LOGIN
# ============================================================

user = get_current_user()

if not user:
    st.markdown(
        """
        <div style="
            max-width:620px;
            margin:9vh auto 20px;
            text-align:center;
            padding:34px;
            border:1px solid #1c4a84;
            border-radius:22px;
            background:linear-gradient(145deg,rgba(7,31,70,.95),rgba(4,20,45,.95));
            box-shadow:0 20px 60px rgba(0,0,0,.25);
        ">
            <div style="font-size:3.5rem;">⚖️</div>
            <div style="font-size:1.9rem;font-weight:850;">Assistente Jurídico IA</div>
            <div style="color:#8fa9c9;margin-top:8px;">
                Inteligência artificial para documentos, processos,
                riscos e análises jurídicas.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login"):
        email = st.text_input("E-mail", "admin@demo.local")
        password = st.text_input("Senha", "admin123", type="password")
        submitted = st.form_submit_button(
            "Entrar no sistema",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            if authenticate(email, password):
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
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

page_options = [
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
    "Perfil",
]


# ============================================================
# SIDEBAR PREMIUM
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="legal-brand">
            <div class="legal-brand-row">
                <div class="legal-logo">⚖️</div>
                <div>
                    <div class="legal-title">Assistente Jurídico IA</div>
                    <div class="legal-sub">Inteligência Artificial v3.1</div>
                </div>
            </div>
        </div>

        <div class="nav-group">PRINCIPAL</div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("Dashboard", "🏠"),
        ("Assistente IA", "🤖"),
        ("Documentos", "📄"),
        ("Processos", "⚖️"),
        ("Riscos", "🛡️"),
        ("Prazos", "📅"),
        ("Relatórios", "📊"),
        ("Base de Conhecimento", "🗄️"),
        ("Configurações", "⚙️"),
        ("Auditoria", "🛡️"),
        ("Perfil", "👤"),
    ]

    page = st.session_state.page
    for nav_name, nav_icon in nav_items:
        active = "sidebar-active" if page == nav_name else ""
        st.markdown(f'<div class="{active}">', unsafe_allow_html=True)
        if st.button(
            f"{nav_icon}  {nav_name}",
            key=f"nav_{nav_name}",
            use_container_width=True,
        ):
            st.session_state.page = nav_name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.page = page

    st.markdown(
        """
        <div class="nav-group">SISTEMA</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  Nova conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.page = "Assistente IA"
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-profile">
            <div class="profile-row">
                <div class="avatar">👤</div>
                <div>
                    <div class="profile-name">Dr. Sérgio Luiz</div>
                    <div class="profile-role">Administrador</div>
                </div>
            </div>
            <div class="online">● Sistema Online</div>
            <div class="version">v3.1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🧹  Atualizar conexão IA", use_container_width=True):
        try:
            clear_ai_cache()
            st.rerun()
        except Exception as exc:
            st.error(f"Não foi possível atualizar a conexão: {exc}")

    if st.button("🚪  Sair do sistema", use_container_width=True):
        audit(action="logout")
        logout()
        st.rerun()


# ============================================================
# TOP BAR
# ============================================================

top1, top2, top3, top4 = st.columns([7, .8, 1.25, 1.5])

with top1:
    st.markdown('<div class="top-search-wrap">', unsafe_allow_html=True)
    global_search = st.text_input(
        "Pesquisa global",
        placeholder="🔍  Buscar processos, documentos, clientes...",
        key="global_search",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with top2:
    st.markdown(
        '<div style="text-align:center;font-size:1.2rem;padding-top:5px;">🔔 <span style="font-size:.62rem;color:#ff5a72;">3</span></div>',
        unsafe_allow_html=True,
    )

with top3:
    st.markdown(
        '<div style="color:#a9c0df;font-size:.67rem;text-align:center;padding-top:2px;">09 de Setembro de 2026<br><b style="color:#fff">12:48</b></div>',
        unsafe_allow_html=True,
    )

with top4:
    st.markdown(
        '<div style="text-align:right;color:#d9e8fb;font-size:.73rem;padding-top:6px;">👤 <b>Dr. Sérgio Luiz</b>⌄</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ============================================================
# PESQUISA GLOBAL
# ============================================================

if global_search and global_search.strip():
    search_term = global_search.strip()
    org_id = user.get("organization_id")
    case_results = []
    doc_results = []

    try:
        case_results = search_cases(org_id, search_term, limit=8) or []
    except Exception:
        case_results = []

    try:
        all_docs = list_documents(org_id) or []
        term_lower = search_term.lower()
        doc_results = [
            d for d in all_docs
            if term_lower in str(d.get("name", d.get("filename", ""))).lower()
        ][:8]
    except Exception:
        doc_results = []

    st.markdown(
        f'<div class="search-results"><b>🔎 Resultados para:</b> {search_term}',
        unsafe_allow_html=True,
    )

    if case_results:
        st.markdown("**⚖️ Processos**", unsafe_allow_html=True)
        for item in case_results:
            title = item.get("title", "Processo")
            client = item.get("client", "Cliente não informado")
            st.markdown(
                f'<div class="search-result-item">⚖️ <b>{title}</b><br>'
                f'<span style="color:#8eabd0">Cliente: {client} · Status: {item.get("status","N/D")}</span></div>',
                unsafe_allow_html=True,
            )

    if doc_results:
        st.markdown("**📄 Documentos**", unsafe_allow_html=True)
        for item in doc_results:
            name = item.get("name", item.get("filename", "Documento"))
            st.markdown(
                f'<div class="search-result-item">📄 <b>{name}</b><br>'
                f'<span style="color:#8eabd0">Status: {item.get("status","N/D")}</span></div>',
                unsafe_allow_html=True,
            )

    if not case_results and not doc_results:
        st.info("Nenhum processo ou documento encontrado para essa pesquisa.")

    st.markdown("</div>", unsafe_allow_html=True)



# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    org_id = user.get("organization_id")
    total_docs, total_cases = get_counts(org_id)

    # valores reais quando disponíveis; não fabricamos números do banco
    docs_value = total_docs
    cases_value = total_cases

    st.markdown(
        """
        <div class="page-title">Bom dia, Dr. Sérgio Luiz 👋</div>
        <div class="page-subtitle">
            Aqui está o resumo da sua operação jurídica hoje.
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card(
            "📄", "Processos Ativos", cases_value,
            "↑ Operação monitorada", "dados atuais do banco", "kpi-blue"
        )
    with k2:
        metric_card(
            "📑", "Documentos", docs_value,
            "↑ Base jurídica", "documentos cadastrados", "kpi-purple"
        )
    with k3:
        metric_card(
            "🛡️", "Riscos Identificados", "—",
            "Aguardando análise", "sem inventar métricas", "kpi-red"
        )
    with k4:
        metric_card(
            "🎯", "Análises de IA", "—",
            "Pipeline disponível", "consulte o Assistente IA", "kpi-teal"
        )

    st.markdown("<div style='height:13px'></div>", unsafe_allow_html=True)

    # esquerda / direita
    left, right = st.columns([1.9, 1.1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("📊", "Evolução dos Casos", "Visão operacional")
        st.plotly_chart(
            plot_dark_line(),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:13px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            section_header("📈", "Processos por Status", "Distribuição atual")
            st.plotly_chart(
                plot_status_donut(),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                """
                <div style="font-size:.72rem;color:#9ab0ce;line-height:1.9">
                🔵 Em andamento<br>
                🟣 Em análise<br>
                🟠 Pendente<br>
                ⚪ Arquivado
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            section_header("📄", "Documentos Recentes", "Base jurídica")
            try:
                docs = list_documents(org_id) or []
            except Exception:
                docs = []

            if docs:
                for doc in docs[:5]:
                    name = str(doc.get("name", "Documento"))
                    status = str(doc.get("status", ""))
                    icon = "📕" if name.lower().endswith(".pdf") else "📘"
                    st.markdown(
                        f"""
                        <div class="activity">
                            <span class="activity-icon">{icon}</span>
                            <span style="font-size:.73rem">{name[:36]}</span>
                            <span style="float:right;color:#6faeff;font-size:.64rem">{status}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Nenhum documento cadastrado ainda.")
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="ai-online">Online</span>
            <div style="font-size:1.05rem;font-weight:800;">🤖 Assistente Jurídico IA</div>
            <div style="font-size:.7rem;color:#8eadd2;margin-top:4px;">Como posso ajudar você hoje?</div>
            """,
            unsafe_allow_html=True,
        )

        quick_q = st.text_input(
            "Pergunta",
            placeholder="Digite sua pergunta...",
            label_visibility="collapsed",
            key="dashboard_ai_question",
        )

        if st.button("➤  Consultar IA", type="primary", use_container_width=True):
            if quick_q.strip():
                st.session_state.pending_question = quick_q.strip()
                st.session_state.page = "Assistente IA"
                st.rerun()

        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("🔎 Consultar RAG", use_container_width=True):
                st.session_state.pending_question = "Consulte a base jurídica e apresente as evidências relevantes."
                st.session_state.page = "Assistente IA"
                st.rerun()
        with qa2:
            if st.button("📄 Analisar documento", use_container_width=True):
                st.session_state.pending_question = "Analise o documento selecionado de forma crítica."
                st.session_state.page = "Assistente IA"
                st.rerun()

        qa3, qa4 = st.columns(2)
        with qa3:
            if st.button("🛡️ Avaliar risco", use_container_width=True):
                st.session_state.pending_question = "Identifique os principais riscos jurídicos."
                st.session_state.page = "Assistente IA"
                st.rerun()
        with qa4:
            if st.button("📋 Resumir processo", use_container_width=True):
                st.session_state.pending_question = "Gere um resumo executivo do processo."
                st.session_state.page = "Assistente IA"
                st.rerun()

        st.markdown(
            """
            <div style="
                margin-top:14px;
                padding:16px;
                border-radius:12px;
                background:rgba(7,32,72,.65);
                border:1px solid rgba(67,115,180,.35);
                text-align:center;
                color:#b8cae2;
                font-size:.72rem;
                font-style:italic;
            ">
                ✨ “A informação certa, no momento certo,<br>
                faz toda a diferença.”
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:13px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("🕘", "Atividade Recente", "Eventos do sistema")

        activities = [
            ("📄", "Análise de documento", "Pipeline de IA disponível"),
            ("⚖️", "Novo processo", "Registro jurídico"),
            ("🛡️", "Análise de risco", "Consulte o Assistente"),
            ("📋", "Tarefa", "Acompanhamento operacional"),
        ]

        for icon, title, desc in activities:
            st.markdown(
                f"""
                <div class="activity">
                    <span class="activity-icon">{icon}</span>
                    <b style="font-size:.72rem">{title}</b><br>
                    <span style="margin-left:44px;color:#7895b8;font-size:.64rem">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-banner">
            ✨ &nbsp; Seu aliado na tomada de decisões jurídicas.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ASSISTENTE IA
# ============================================================

elif page == "Assistente IA":
    st.markdown(
        """
        <div class="page-title">🤖 Assistente Jurídico IA</div>
        <div class="page-subtitle">
            RAG + Retriever + Reranker + Agentes + Evidências.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        llm_status = ai_status()
    except Exception as exc:
        llm_status = {
            "status": "error",
            "provider": "indisponível",
            "model": "N/D",
            "configured": False,
            "error": str(exc),
        }

    status_value = str(llm_status.get("status", "unknown"))
    provider_value = str(llm_status.get("provider", "N/D"))
    model_value = str(llm_status.get("model", "N/D"))

    if status_value == "connected":
        status_html = f'<span class="ai-status-ok">● IA conectada</span> · {provider_value} · {model_value}'
    elif status_value == "demo":
        status_html = '<span class="ai-status-warn">● Modo demonstração</span> · configure o provedor LLM no Streamlit'
    else:
        status_html = f'<span class="ai-status-error">● IA não configurada</span> · {provider_value}'

    st.markdown(
        f'<div class="ai-status-card">🤖 <b>Status do Assistente</b><br>'
        f'<span style="color:#9db6d8;font-size:.82rem">{status_html}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            selected_agent = st.selectbox(
                "Agente",
                [
                    "⚖️ Agente Jurídico",
                    "⚠️ Agente de Risco",
                    "📝 Agente de Resumo",
                    "💬 Agente Geral",
                    "🔎 RAG / Base Jurídica",
                ],
                key="sel_agent",
            )

        with c2:
            selected_mode = st.selectbox(
                "Modo de análise",
                [
                    "Análise jurídica completa",
                    "Verificação de conformidade",
                    "Auditoria de cláusulas",
                    "Busca jurisprudencial",
                ],
                key="sel_mode",
            )

        with c3:
            selected_depth = st.selectbox(
                "Profundidade",
                ["Detalhado", "Resumido", "Executivo", "Avançado (RAG estendido)"],
                key="sel_depth",
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    chat_col, info_col = st.columns([2.1, 1])

    with chat_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("💬", "Consulta jurídica", "Faça uma pergunta para o pipeline de IA")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Pergunte sobre processos, documentos, riscos ou evidências...")

        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = None

        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            mode_map = {
                "⚠️ Agente de Risco": "risk",
                "📝 Agente de Resumo": "summary",
                "💬 Agente Geral": "general",
                "⚖️ Agente Jurídico": "legal",
                "🔎 RAG / Base Jurídica": "auto",
            }

            with st.chat_message("assistant"):
                with st.spinner("Executando RAG → Retriever → Reranker → Agente IA..."):
                    result = call_orchestrator(
                        query=question,
                        org_id=user.get("organization_id"),
                        mode=mode_map.get(selected_agent, "auto"),
                        top_k=8,
                        rerank_k=5,
                    )

                answer = str(result.get("answer", "") or "").strip()
                provider_error = result.get("error")

                if not answer:
                    answer = "Não foi possível gerar uma resposta."

                st.markdown(answer)

                if provider_error:
                    st.warning(f"Detalhe técnico do Assistente: {provider_error}")

                if result.get("agent_status") == "error" and not provider_error:
                    primary = result.get("primary") or {}
                    primary_error = primary.get("error") if isinstance(primary, dict) else None
                    if primary_error:
                        st.warning(f"Falha no agente: {primary_error}")

                citations = result.get("citations") or []
                if citations:
                    st.markdown("#### 📚 Evidências e citações")
                    for i, citation in enumerate(citations, 1):
                        if not isinstance(citation, dict):
                            continue
                        doc = citation.get("document", "Documento")
                        page_no = citation.get("page", "N/D")
                        content = citation.get("content", citation.get("text", ""))
                        st.markdown(
                            f"""
                            <div class="glass-card" style="margin-bottom:8px;padding:12px">
                                <b>[{citation.get("id", i)}] {doc}</b>
                                <span style="color:#7fa3cd"> · Página {page_no}</span>
                                <div style="margin-top:6px;color:#a7bdd9;font-size:.76rem">
                                    {content}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
        section_header("🛡️", "Segurança da resposta", "Guard Agent / avaliação")
        st.markdown(
            """
            <span class="badge badge-green">● Pipeline ativo</span>
            <br><br>
            <span class="badge badge-blue">RAG</span>
            <span class="badge badge-blue">Reranker</span>
            <span class="badge badge-blue">Citações</span>
            <span class="badge badge-blue">Guard</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧹 Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DOCUMENTOS
# ============================================================

elif page == "Documentos":
    org_id = user.get("organization_id")

    st.markdown(
        """
        <div class="page-title">📄 Documentos</div>
        <div class="page-subtitle">
            Gestão documental e indexação para o RAG.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        documents = list_documents(org_id) or []
    except Exception:
        documents = []

    total_docs = len(documents)
    ready_docs = sum(1 for d in documents if d.get("status") == "Indexado")

    a, b, c, d = st.columns(4)
    for col, icon, label, value in [
        (a, "📄", "Documentos", total_docs),
        (b, "📚", "Páginas", sum(int(x.get("pages", 0) or 0) for x in documents)),
        (c, "🧩", "Chunks", sum(int(x.get("chunks", 0) or 0) for x in documents)),
        (d, "🟢", "Prontos para RAG", ready_docs),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="kpi kpi-blue">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:13px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.65, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("📤", "Adicionar documento", "PDF, DOCX ou TXT")

        uploaded_file = st.file_uploader(
            "Carregar arquivo",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )

        use_ocr = st.checkbox(
            "Usar OCR quando necessário",
            value=True,
        )

        if uploaded_file and st.button(
            "⚡ Processar e indexar documento",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.status(
                    "Processando documento...",
                    expanded=True,
                ) as status:
                    result = ingest_document(
                        uploaded_file,
                        org_id,
                        use_ocr=use_ocr,
                    )
                    st.write(f"✓ {result.get('pages', 0)} página(s)")
                    st.write(f"✓ {result.get('chunks', 0)} chunk(s)")
                    st.write(f"✓ {result.get('indexed_chunks', 0)} chunk(s) indexado(s)")
                    status.update(
                        label="Documento pronto para consulta via RAG",
                        state="complete",
                    )
                audit(
                    action="document_upload",
                    details={"filename": uploaded_file.name, "result": result},
                    organization_id=org_id,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Falha na indexação: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("🧠", "Motor RAG", "Status do conhecimento")
        st.markdown(
            f"""
            <div style="line-height:2;color:#9ab0ce;font-size:.78rem">
            <b>Vector Store:</b> FAISS<br>
            <b>Embeddings:</b> Sentence Transformers<br>
            <b>Reranker:</b> CrossEncoder<br>
            <b>Status:</b> <span class="badge badge-green">Operacional</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:13px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("🗂️", "Biblioteca de documentos", "Arquivos da organização")

    search_doc = st.text_input(
        "Buscar documento",
        placeholder="Digite o nome do arquivo...",
        label_visibility="collapsed",
    )

    visible = documents
    if search_doc:
        visible = [
            x for x in visible
            if search_doc.lower() in str(x.get("name", "")).lower()
        ]

    if not visible:
        st.info("Nenhum documento encontrado.")
    else:
        for doc in visible:
            name = doc.get("name", "Documento")
            status = doc.get("status", "Desconhecido")
            status_cls = (
                "badge-green" if status == "Indexado"
                else "badge-orange" if status == "Processando"
                else "badge-red"
            )
            c1, c2, c3 = st.columns([5, 1.5, 1])
            with c1:
                st.markdown(f"📄 **{name}**")
            with c2:
                st.markdown(
                    f'<span class="badge {status_cls}">{status}</span>',
                    unsafe_allow_html=True,
                )
            with c3:
                if st.button(
                    "🤖",
                    key=f"analyze_doc_{doc.get('id')}",
                    help="Analisar com IA",
                ):
                    st.session_state.pending_question = (
                        f"Analise detalhadamente o documento {name}, "
                        "identifique riscos, obrigações e pontos críticos."
                    )
                    st.session_state.page = "Assistente IA"
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PROCESSOS
# ============================================================

elif page == "Processos":
    org_id = user.get("organization_id")

    st.markdown(
        """
        <div class="page-title">⚖️ Processos</div>
        <div class="page-subtitle">
            Central de casos jurídicos da organização.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_process_id = st.session_state.get("active_process_id")

    if selected_process_id:
        case = get_case(org_id, selected_process_id)

        if st.button("← Voltar para processos"):
            st.session_state.pop("active_process_id", None)
            st.rerun()

        if not case:
            st.error("Processo não encontrado.")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### ⚖️ {case.get('title', 'Processo')}")
            st.markdown(
                f"""
                **Cliente:** {case.get('client', '—')}  
                **Categoria:** {case.get('category', '—')}  
                **Prioridade:** {case.get('priority', '—')}  
                **Status:** {case.get('status', '—')}
                """,
            )

            if st.button(
                "🤖 Analisar processo com IA",
                type="primary",
            ):
                st.session_state.pending_question = (
                    f"Faça uma análise jurídica completa do processo "
                    f"{case.get('title')}, considerando cliente, categoria e prioridade."
                )
                st.session_state.page = "Assistente IA"
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            new_status = st.selectbox(
                "Atualizar status",
                CASE_STATUSES,
                index=(
                    CASE_STATUSES.index(case.get("status"))
                    if case.get("status") in CASE_STATUSES else 0
                ),
            )

            if st.button("Salvar status"):
                result = update_case_status(
                    org_id,
                    selected_process_id,
                    new_status,
                )
                if result.get("updated"):
                    audit(
                        action="case_status_update",
                        details={
                            "case_id": selected_process_id,
                            "new_status": new_status,
                        },
                        organization_id=org_id,
                        entity_type="case",
                        entity_id=selected_process_id,
                    )
                    st.success("Status atualizado.")
                    st.rerun()

    else:
        with st.expander("＋ Criar novo processo"):
            with st.form("new_case_form", clear_on_submit=True):
                title = st.text_input("Título do processo")
                client = st.text_input("Cliente")
                category = st.text_input("Categoria")
                priority = st.selectbox("Prioridade", CASE_PRIORITIES)

                if st.form_submit_button(
                    "Criar processo",
                    type="primary",
                ):
                    try:
                        created = create_case(
                            org_id,
                            title,
                            client,
                            category,
                            priority,
                        )
                        audit(
                            action="case_create",
                            details={
                                "case_id": created.get("case_id"),
                                "title": title,
                            },
                            organization_id=org_id,
                        )
                        st.success("Processo criado.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        try:
            cases = list_cases(org_id) or []
        except Exception:
            cases = []

        search = st.text_input(
            "Buscar processo",
            placeholder="Título, cliente ou categoria...",
            label_visibility="collapsed",
        )

        if search:
            try:
                cases = search_cases(org_id, search) or []
            except Exception:
                cases = [
                    x for x in cases
                    if search.lower() in str(x).lower()
                ]

        if not cases:
            st.info("Nenhum processo cadastrado.")
        else:
            for case in cases:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                c1, c2, c3 = st.columns([4, 2, 1])

                with c1:
                    st.markdown(
                        f"### ⚖️ {case.get('title', 'Processo')}"
                    )
                    st.caption(
                        f"{case.get('client', '—')} · "
                        f"{case.get('category', '—')}"
                    )

                with c2:
                    st.markdown(
                        f'<span class="badge badge-blue">{case.get("status", "—")}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Prioridade: {case.get('priority', '—')}")

                with c3:
                    if st.button(
                        "Abrir",
                        key=f"open_case_{case.get('id')}",
                        use_container_width=True,
                    ):
                        st.session_state.active_process_id = case.get("id")
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ============================================================
# RISCOS
# ============================================================

elif page == "Riscos":
    st.markdown(
        """
        <div class="page-title">🛡️ Riscos Jurídicos</div>
        <div class="page-subtitle">
            Central de análise e monitoramento de riscos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="kpi kpi-red">', unsafe_allow_html=True)
        st.markdown("### 🔴 Alto")
        st.markdown("Consulte o Assistente IA")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="kpi kpi-purple">', unsafe_allow_html=True)
        st.markdown("### 🟠 Médio")
        st.markdown("Itens que exigem revisão")
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        st.markdown('<div class="kpi kpi-teal">', unsafe_allow_html=True)
        st.markdown("### 🟢 Controle")
        st.markdown("Sem risco registrado")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("🤖", "Nova análise de risco", "Use o agente especializado")
    q = st.text_area(
        "Contexto",
        placeholder="Descreva o processo, contrato ou situação que deseja avaliar...",
        height=130,
    )
    if st.button("⚡ Executar análise de risco", type="primary"):
        if q.strip():
            with st.spinner("Executando agente de risco..."):
                try:
                    result = risk_analysis(
                        q,
                        organization_id=user.get("organization_id"),
                    )
                except TypeError:
                    try:
                        result = risk_analysis(q, user.get("organization_id"))
                    except Exception as exc:
                        result = {"answer": "", "error": str(exc)}
                except Exception as exc:
                    result = {"answer": "", "error": str(exc)}

            if isinstance(result, dict):
                st.markdown(result.get("answer", "Nenhum resultado retornado."))
            else:
                st.markdown(str(result))
        else:
            st.warning("Informe o contexto da análise.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PRAZOS
# ============================================================

elif page == "Prazos":
    st.markdown(
        """
        <div class="page-title">📅 Prazos</div>
        <div class="page-subtitle">
            Visão operacional para acompanhamento de prazos jurídicos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    items = [
        ("10/09/2026", "Processo #2026-0145", "Manifestação processual", "Alto"),
        ("11/09/2026", "Documento pendente", "Assinatura / revisão", "Médio"),
        ("12/09/2026", "Análise contratual", "Revisão jurídica", "Médio"),
    ]

    for date_, title, desc, priority in items:
        cls = "badge-red" if priority == "Alto" else "badge-orange"
        st.markdown(
            f"""
            <div class="section-card" style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <b>📅 {title}</b>
                        <div style="color:#7895b8;font-size:.72rem;margin-top:4px">{desc}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-weight:800">{date_}</div>
                        <span class="badge {cls}">{priority}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RELATÓRIOS
# ============================================================

elif page == "Relatórios":
    org_id = user.get("organization_id")
    cases, documents, risks, deadlines = load_report_data(org_id)

    st.markdown(
        """
        <div class="page-title">📊 Relatórios</div>
        <div class="page-subtitle">
            Painel executivo com indicadores, filtros, gráficos e exportação.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- FILTROS --------------------
    with st.container(border=True):
        st.markdown("**🔎 Filtros do relatório**")
        f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.3, 1.1])

        case_df_all = report_cases_dataframe(cases)
        categories = sorted([x for x in case_df_all["Categoria"].dropna().astype(str).unique().tolist() if x.strip()]) if not case_df_all.empty else []
        statuses = sorted([x for x in case_df_all["Status"].dropna().astype(str).unique().tolist() if x.strip()]) if not case_df_all.empty else []

        with f1:
            period = st.selectbox("Período", ["Todo o período", "Últimos 7 dias", "Últimos 30 dias", "Últimos 90 dias"], key="report_period")
        with f2:
            category_filter = st.selectbox("Categoria", ["Todas"] + categories, key="report_category")
        with f3:
            status_filter = st.selectbox("Status", ["Todos"] + statuses, key="report_status")
        with f4:
            priority_filter = st.selectbox("Prioridade", ["Todas", "Alta", "Média", "Baixa"], key="report_priority")

    filtered_cases = list(cases)
    if category_filter != "Todas":
        filtered_cases = [c for c in filtered_cases if str(c.get("category", "")) == category_filter]
    if status_filter != "Todos":
        filtered_cases = [c for c in filtered_cases if str(c.get("status", "")) == status_filter]
    if priority_filter != "Todas":
        filtered_cases = [c for c in filtered_cases if str(c.get("priority", "")) == priority_filter]

    if period != "Todo o período":
        days = {"Últimos 7 dias": 7, "Últimos 30 dias": 30, "Últimos 90 dias": 90}[period]
        cutoff = date.today() - timedelta(days=days)
        filtered_cases = [
            c for c in filtered_cases
            if (_to_date(c.get("created_at", c.get("createdAt"))) is None)
            or (_to_date(c.get("created_at", c.get("createdAt"))) >= cutoff)
        ]

    case_df = report_cases_dataframe(filtered_cases)
    total_processes = len(filtered_cases)
    total_documents = len(documents)
    total_risks = len(risks)
    total_deadlines = len(deadlines)

    # -------------------- KPIs --------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("⚖️", "Processos", total_processes, "Base atual", "processos filtrados", "kpi-blue")
    with k2:
        metric_card("📄", "Documentos", total_documents, "Base documental", "arquivos cadastrados", "kpi-purple")
    with k3:
        metric_card("🛡️", "Riscos", total_risks, "Registrados", "na auditoria", "kpi-red")
    with k4:
        metric_card("📅", "Prazos", total_deadlines, "Monitorados", "agenda operacional", "kpi-teal")
    with k5:
        high = sum(1 for c in filtered_cases if str(c.get("priority", "")).lower() == "alta")
        metric_card("🚨", "Alta prioridade", high, "Atenção", "processos filtrados", "kpi-red")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # -------------------- GRÁFICOS --------------------
    g1, g2 = st.columns([1.35, 1])
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("📊", "Processos por categoria", "Distribuição da carteira atual")
        st.plotly_chart(report_category_chart(filtered_cases), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("🧭", "Processos por status", "Situação atual da carteira")
        st.plotly_chart(report_status_chart(filtered_cases), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # -------------------- INDICADORES --------------------
    i1, i2 = st.columns(2)
    with i1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("📌", "Indicadores da operação", "Leitura executiva")
        status_counts = case_df["Status"].value_counts().to_dict() if not case_df.empty else {}
        priority_counts = case_df["Prioridade"].value_counts().to_dict() if not case_df.empty else {}
        st.markdown(
            f"""
            <div class="glass-card" style="padding:14px;margin-bottom:8px"><b>Processos ativos</b><span style="float:right;font-size:1.1rem">{sum(v for k,v in status_counts.items() if k.lower() in ["em andamento","em análise","pendente"])}</span></div>
            <div class="glass-card" style="padding:14px;margin-bottom:8px"><b>Em andamento</b><span style="float:right;font-size:1.1rem">{status_counts.get("Em andamento", 0)}</span></div>
            <div class="glass-card" style="padding:14px;margin-bottom:8px"><b>Alta prioridade</b><span style="float:right;font-size:1.1rem">{priority_counts.get("Alta", 0)}</span></div>
            <div class="glass-card" style="padding:14px"><b>Documentos prontos para RAG</b><span style="float:right;font-size:1.1rem">{sum(1 for d in documents if d.get("status") == "Indexado")}</span></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with i2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("📅", "Prazos e alertas", "Itens que exigem acompanhamento")
        for item in deadlines:
            cls = "badge-red" if item["priority"] == "Alto" else "badge-orange"
            st.markdown(
                f"""
                <div class="activity">
                    <b>📅 {item['title']}</b>
                    <span style="float:right" class="badge {cls}">{item['priority']}</span><br>
                    <span style="color:#7895b8;font-size:.68rem">{item['description']} · {item['date']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # -------------------- TABELA + EXPORTAÇÃO --------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("📋", "Detalhamento dos processos", f"{len(case_df)} registro(s) no filtro atual")
    if case_df.empty:
        st.info("Nenhum processo corresponde aos filtros selecionados.")
    else:
        st.dataframe(case_df, use_container_width=True, hide_index=True)

    high = sum(1 for c in filtered_cases if str(c.get("priority", "")).lower() == "alta")
    summary = {
        "processes": total_processes,
        "documents": total_documents,
        "risks": total_risks,
        "deadlines": total_deadlines,
        "high_priority": high,
    }

    st.markdown("<div style='margin-top:8px;margin-bottom:8px;font-weight:700;font-size:1rem'>📤 Exportar relatório</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    stamp = datetime.now().strftime('%Y%m%d_%H%M')

    with b1:
        st.download_button(
            "📄 Baixar PDF",
            data=build_report_pdf(summary, case_df, total_documents, total_risks, deadlines),
            file_name=f"relatorio_juridico_{stamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_report_pdf",
        )

    with b2:
        st.download_button(
            "📊 Baixar Excel",
            data=build_report_excel(summary, case_df, documents, risks, deadlines),
            file_name=f"relatorio_juridico_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_report_excel",
        )

    with b3:
        st.download_button(
            "📋 Baixar CSV",
            data=build_report_csv(filtered_cases, documents, risks, deadlines),
            file_name=f"relatorio_juridico_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_report_csv",
        )

    st.markdown(
        "<div style='padding:10px 12px;color:#89a7cc;font-size:.70rem'>💡 PDF para apresentação, Excel para análise e CSV para integração. Os arquivos usam os dados disponíveis na organização.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# BASE DE CONHECIMENTO
# ============================================================

elif page == "Base de Conhecimento":
    st.markdown(
        """
        <div class="page-title">🗄️ Base de Conhecimento</div>
        <div class="page-subtitle">
            Conteúdo indexado utilizado pelo RAG.
        </div>
        """,
        unsafe_allow_html=True,
    )

    docs, cases = get_counts(user.get("organization_id"))

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("📄", "Documentos", docs, "Base disponível", "organização", "kpi-blue")
    with c2:
        metric_card("⚖️", "Processos", cases, "Contexto", "organização", "kpi-purple")
    with c3:
        metric_card("🧠", "RAG", "ON", "FAISS", "motor de busca semântica", "kpi-teal")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Pipeline de conhecimento</div>
            <div style="color:#8da8ca;font-size:.76rem;line-height:2;margin-top:10px">
                📄 Documento → OCR / extração → Chunking → Embeddings →
                FAISS → Retriever → Reranker → LLM → Citações
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CONFIGURAÇÕES
# ============================================================

elif page == "Configurações":
    st.markdown(
        """
        <div class="page-title">⚙️ Configurações</div>
        <div class="page-subtitle">
            Preferências do Assistente Jurídico.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("settings_form"):
        provider = st.selectbox(
            "Provedor de IA",
            ["Gemini", "OpenAI"],
        )
        temperature = st.slider(
            "Temperatura",
            0.0,
            1.0,
            0.2,
            0.05,
        )
        top_k = st.number_input(
            "Top-K do RAG",
            min_value=1,
            max_value=30,
            value=8,
        )
        if st.form_submit_button("Salvar configurações", type="primary"):
            st.success("Preferências atualizadas para esta sessão.")


# ============================================================
# AUDITORIA
# ============================================================

elif page == "Auditoria":
    st.markdown(
        """
        <div class="page-title">🛡️ Auditoria</div>
        <div class="page-subtitle">
            Rastreamento das ações realizadas no sistema.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT action, details, created_at
                FROM audit_logs
                ORDER BY created_at DESC
                LIMIT 100
                """
            ).fetchall()

        if rows:
            data = [
                {
                    "Ação": row[0],
                    "Detalhes": row[1],
                    "Data": row[2],
                }
                for row in rows
            ]
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Nenhum evento de auditoria encontrado.")
    except Exception as exc:
        st.warning(f"Não foi possível carregar a auditoria: {exc}")


# ============================================================
# PERFIL
# ============================================================

elif page == "Perfil":
    st.markdown(
        """
        <div class="page-title">👤 Perfil</div>
        <div class="page-subtitle">
            Informações do usuário atual.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="glass-card" style="max-width:760px">
            <div style="display:flex;gap:16px;align-items:center">
                <div class="avatar" style="width:58px;height:58px;font-size:27px">👤</div>
                <div>
                    <div style="font-size:1.15rem;font-weight:800">
                        {user.get("name", "Usuário Jurídico")}
                    </div>
                    <div style="color:#8fa9c9;font-size:.75rem">
                        {user.get("email", "—")}
                    </div>
                </div>
            </div>
            <hr style="border-color:#173e73;margin:18px 0">
            <div style="color:#8fa9c9;font-size:.75rem">
                Organização: <b style="color:#fff">{user.get("organization_id", "—")}</b>
            </div>
            <div style="color:#8fa9c9;font-size:.75rem;margin-top:7px">
                Status: <span class="badge badge-green">Ativo</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#58789f;
        font-size:.64rem;
        margin-top:20px;
        padding-top:10px;
        border-top:1px solid rgba(40,83,133,.28);
    ">
        ⚖️ Assistente Jurídico IA · RAG + Multiagentes + Evidências · V3.1
    </div>
    """,
    unsafe_allow_html=True,
)
