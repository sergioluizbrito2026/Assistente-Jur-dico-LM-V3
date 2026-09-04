from __future__ import annotations

import inspect
import traceback
from typing import Any, Dict

import streamlit as st

# ============================================================
# BANCO DE DADOS
# ============================================================

from db import init_db, seed_demo

# ============================================================
# SERVIÇOS
# ============================================================

from services.audit import audit
from services.auth import authenticate, get_current_user, logout
from services.cases import create_case, list_cases
from services.documents import list_documents
from services.evaluation import evaluate_answer
from services.ingestion import ingest_document
from services.rag_pipeline import rag_answer, retrieve_and_rerank
from services.ai_orchestrator import orchestrate, risk_analysis


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

try:
    init_db()

except Exception as exc:

    st.error(
        "Erro ao inicializar o banco de dados."
    )

    with st.expander("Detalhes técnicos"):

        st.code(
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
        )

    st.stop()


try:
    seed_demo()

except Exception as exc:

    st.warning(
        "O banco foi inicializado, mas os dados "
        "de demonstração não puderam ser carregados."
    )

    with st.expander("Detalhes do seed"):

        st.code(
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
        )

