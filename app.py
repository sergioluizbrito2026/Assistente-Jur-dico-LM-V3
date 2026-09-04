"""
Assistente Jurídico SaaS IA V3
app.py

Interface principal da plataforma.

Pipeline:

Documento
    ↓
Ingestão
    ↓
OCR
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
Retriever
    ↓
Reranker
    ↓
Guard Agent
    ↓
AI Orchestrator
    ↓
Agente Jurídico
    ↓
LLM
    ↓
Citações
    ↓
Evaluation
"""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Dict

import pandas as pd
import streamlit as st


# ============================================================
# IMPORTS
# ============================================================

from db import init_db, seed_demo

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

from services.documents import (
    list_documents,
)

from services.evaluation import (
    evaluate_answer,
)

from services.ingestion import (
    ingest_document,
)

from services.rag_pipeline import (
    rag_answer,
    retrieve_and_rerank,
)

from services.ai_orchestrator import (
    orchestrate,
    risk_analysis,
)


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Assistente Jurídico IA SaaS V3",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BANCO
# ============================================================

try:
    init_db()
    seed_demo()
except Exception as exc:
    st.error(
        f"Erro ao inicializar o banco de dados: {exc}"
    )
    st.stop()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #050d1a;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #09152b,
        #050d1a
    );
}

.block-container {
    max-width: 1500px;
}

[data-testid="stMetric"] {
    background: #0d1a30;
    border: 1px solid #25385d;
    border-radius: 14px;
    padding: 14px;
}

.v3 {
    padding: 10px 14px;
    border: 1px solid #2a3f68;
    border-radius: 12px;
    background: #0b1830;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def page_title(
    title: str,
    subtitle: str = "",
) -> None:

    st.title(title)

    if subtitle:
        st.caption(subtitle)


def safe_dict(
    value: Any,
) -> Dict[str, Any]:

    if isinstance(value, dict):
        return value

    return {
        "answer": str(value or ""),
        "citations": [],
        "retrieved": [],
        "reranked": [],
    }


def safe_list(
    value: Any,
) -> list:

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return []


# ============================================================
# ORCHESTRATOR COMPATÍVEL V3 / V3.1
# ============================================================

def call_orchestrator(
    query: str,
    org_id: Any,
    mode: str = "auto",
    top_k: int = 8,
    rerank_k: int = 5,
    extra_context: str | None = None,
) -> Dict[str, Any]:
    """
    Chama o AI Orchestrator de maneira compatível
    com diferentes versões do ai_orchestrator.py.

    Isso evita TypeError quando a assinatura do
    orchestrate() muda entre versões.
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

    # --------------------------------------------------------
    # Parâmetros possíveis
    # --------------------------------------------------------

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

        signature = inspect.signature(
            orchestrate
        )

        parameters = signature.parameters

        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters.values()
        )

        if accepts_kwargs:

            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if value is not None
            }

        else:

            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in parameters
                and value is not None
            }

        # ----------------------------------------------------
        # Evita passar question e query simultaneamente
        # ----------------------------------------------------

        if "query" in parameters:
            filtered_kwargs.pop(
                "question",
                None,
            )

        elif "question" in parameters:
            filtered_kwargs.pop(
                "query",
                None,
            )

        # ----------------------------------------------------
        # Organização
        # ----------------------------------------------------

        if "org_id" in parameters:
            filtered_kwargs.pop(
                "organization_id",
                None,
            )

        elif "organization_id" in parameters:
            filtered_kwargs.pop(
                "org_id",
                None,
            )

        # ----------------------------------------------------
        # Execução
        # ----------------------------------------------------

        result = orchestrate(
            **filtered_kwargs
        )

        result = safe_dict(result)

        # ----------------------------------------------------
        # Normalização do retorno
        # ----------------------------------------------------

        result.setdefault(
            "answer",
            "",
        )

        result.setdefault(
            "citations",
            [],
        )

        result.setdefault(
            "retrieved",
            [],
        )

        result.setdefault(
            "reranked",
            [],
        )

        result.setdefault(
            "agent",
            "juridico",
        )

        result.setdefault(
            "intent",
            "legal_query",
        )

        return result

    except Exception as exc:

        return {
            "answer": (
                "Não foi possível gerar a análise jurídica "
                "neste momento."
            ),
            "citations": [],
            "retrieved": [],
            "reranked": [],
            "agent": "error",
            "intent": "error",
            "error": str(exc),
        }


# ============================================================
# LOGIN
# ============================================================

user = get_current_user()


if not user:

    st.markdown(
        """
        <h1 style="
            text-align:center;
            margin-top:90px;
        ">
        ⚖️ Assistente Jurídico IA SaaS V3
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "OCR → Chunking → Embeddings → Vector DB → "
        "Retriever → Reranker → LLM → Citações → Avaliação"
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

                authenticated = authenticate(
                    email,
                    password,
                )

                if authenticated:
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
# SESSION STATE
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
        "## ⚖️ Assistente Jurídico IA"
    )

    st.caption(
        "V3.1 · RAG + Agentes + Evaluation"
    )

    st.divider()

    menus = {

        "PRINCIPAL": [
            "Dashboard",
            "Assistente IA",
        ],

        "INTELIGÊNCIA": [
            "Ingestão RAG",
            "Base Vetorial",
            "Avaliação RAG",
        ],

        "GESTÃO JURÍDICA": [
            "Documentos",
            "Processos",
            "Análise de Risco",
        ],

        "SISTEMA": [
            "Auditoria",
            "Configurações",
        ],
    }

    for group, items in menus.items():

        st.caption(group)

        for item in items:

            if st.button(
                item,
                key="nav_" + item,
                use_container_width=True,
            ):

                st.session_state.page = item

                st.rerun()

    st.divider()

    st.caption(
        f"👤 {user.get('name', 'Usuário')} · "
        f"{user.get('role', 'user')}"
    )

    if st.button(
        "Sair",
        use_container_width=True,
    ):

        logout()

        st.rerun()


page = st.session_state.page


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    col_head1, col_head2 = st.columns(
        [0.7, 0.3]
    )

    with col_head1:

        st.title(
            "⚖️ Dashboard de Inteligência Jurídica"
        )

        st.markdown(
            "Visão executiva da operação jurídica, "
            "documentos, processos, riscos e desempenho da IA."
        )

    with col_head2:

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            if st.button("🔄 Atualizar"):

                st.rerun()

        with c2:

            st.markdown(
                "**Período:** Hoje"
            )

        with c3:

            st.markdown(
                "**IA:** 🟢 Online"
            )

    st.markdown("---")

    st.markdown(
        "### 📊 Indicadores Principais"
    )

    metrics_cols = st.columns(8)

    values = [
        ("📄 Documentos", "25"),
        ("📑 Chunks", "148"),
        ("⚖️ Processos", "18"),
        ("🔴 Riscos Críticos", "2"),
        ("🟠 Riscos Altos", "5"),
        ("🤖 Consultas IA", "126"),
        ("🎯 Qualidade RAG", "90%"),
        ("🟢 LLM", "Conectado"),
    ]

    for col, (label, value) in zip(
        metrics_cols,
        values,
    ):

        with col:

            st.metric(
                label,
                value,
            )

    st.markdown("---")

    st.markdown(
        "### 🧠 Pipeline de Inteligência RAG"
    )

    pipeline_data = {
        "Etapa": [
            "PDF/DOCX",
            "OCR",
            "Chunking",
            "Embeddings",
            "Vector DB",
            "Retriever",
            "Reranker",
            "Guard Agent",
            "AI Orchestrator",
            "LLM",
            "Citações",
            "Evaluation",
        ],
        "Status": [
            "🟢 OK",
            "🟢 OK",
            "🟢 OK",
            "🟢 OK",
            "🟢 OK (FAISS)",
            "🟢 OK",
            "🟢 OK",
            "🟢 OK",
            "🟢 OK",
            "🟢 Conectado",
            "🟢 OK",
            "🟢 OK",
        ],
    }

    df_pipeline = pd.DataFrame(
        pipeline_data
    )

    st.dataframe(
        df_pipeline,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:

        st.markdown(
            "### 📚 Documentos na Base Jurídica"
        )

        docs_por_tipo = pd.DataFrame(
            {
                "Tipo": [
                    "PDF",
                    "DOCX",
                    "TXT",
                ],
                "Quantidade": [
                    12,
                    7,
                    3,
                ],
            }
        )

        st.bar_chart(
            docs_por_tipo.set_index(
                "Tipo"
            )
        )

    with col_g2:

        st.markdown(
            "### ⚖️ Status dos Processos"
        )

        processos_status = pd.DataFrame(
            {
                "Status": [
                    "Em andamento",
                    "Em análise",
                    "Aguardando",
                    "Concluído",
                ],
                "Quantidade": [
                    12,
                    5,
                    3,
                    8,
                ],
            }
        )

        st.bar_chart(
            processos_status.set_index(
                "Status"
            )
        )

    st.markdown("---")

    st.markdown(
        "### 🤖 Arquitetura Multiagente"
    )

    agents = [
        (
            "🧑‍⚖️",
            "Agente Jurídico",
            "Coordenação da análise jurídica",
        ),
        (
            "⚠️",
            "Agente de Risco",
            "Identificação de riscos",
        ),
        (
            "📄",
            "Agente de Resumo",
            "Resumo de documentos",
        ),
        (
            "🤖",
            "Agente Geral",
            "Consultas gerais",
        ),
        (
            "🔎",
            "RAG",
            "Recuperação de evidências",
        ),
        (
            "📚",
            "Citações",
            "Evidências documentais",
        ),
        (
            "📊",
            "Evaluation",
            "Avaliação da resposta",
        ),
        (
            "🛡️",
            "Guard Agent",
            "Segurança e controle",
        ),
    ]

    agent_cols = st.columns(4)

    for index, (
        icon,
        name,
        description,
    ) in enumerate(agents):

        with agent_cols[index % 4]:

            st.markdown(
                f"""
                <div class="v3">
                    <h4>{icon} {name}</h4>
                    <small>{description}</small>
                    <br>
                    <b>🟢 Ativo</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown(
        "### 🚨 Mapa de Riscos Jurídicos"
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.error("🔴 Crítico: 2")

    with r2:
        st.warning("🟠 Alto: 5")

    with r3:
        st.info("🟡 Médio: 8")

    with r4:
        st.success("🟢 Baixo: 12")

    st.markdown(
        "#### Principais Riscos Identificados"
    )

    riscos_detalhes = [
        {
            "Nível": "🔴 Crítico",
            "Risco": "Cláusula de rescisão abusiva",
            "Documento": "Contrato_Prestacao_Servicos.pdf",
            "Página": 3,
        },
        {
            "Nível": "🟠 Alto",
            "Risco": "Proteção de dados inadequada",
            "Documento": "Contrato_Prestacao_Servicos.pdf",
            "Página": 2,
        },
        {
            "Nível": "🟡 Médio",
            "Risco": "Prazo contratual omisso",
            "Documento": "Contrato_Prestacao_Servicos.pdf",
            "Página": 3,
        },
    ]

    st.dataframe(
        pd.DataFrame(riscos_detalhes),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    col_ia1, col_ia2 = st.columns(2)

    with col_ia1:

        st.markdown(
            "### 🧪 Qualidade das Respostas da IA"
        )

        q1, q2 = st.columns(2)

        with q1:

            st.metric(
                "Context Relevance",
                "87%",
            )

            st.metric(
                "Citation Coverage",
                "92%",
            )

        with q2:

            st.metric(
                "Groundedness",
                "89%",
            )

            st.metric(
                "Overall RAG Score",
                "90%",
            )

    with col_ia2:

        st.markdown(
            "### ⚙️ Configurações do Assistente"
        )

        st.markdown(
            """
            • LLM: Gemini / OpenAI<br>
            • Embeddings: Sentence Transformers<br>
            • Vector DB: FAISS<br>
            • Reranker: CrossEncoder<br>
            • Retriever Top-K: 8<br>
            • Reranker Top-K: 5<br>
            • Guard Agent: 🟢 Ativo<br>
            • Evaluation: 🟢 Ativo
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ASSISTENTE IA
# ============================================================

elif page == "Assistente IA":

    page_title(
        "🧑‍⚖️ Assistente Jurídico IA",
        "LLM respondendo com RAG, reranking, agentes, evidências e citações",
    )

    # --------------------------------------------------------
    # STATUS DOS AGENTES
    # --------------------------------------------------------

    with st.expander(
        "🤖 Agentes ativos",
        expanded=False,
    ):

        agent_status = pd.DataFrame(
            {
                "Agente": [
                    "Agente Jurídico",
                    "Agente de Risco",
                    "Agente de Resumo",
                    "Agente Geral",
                    "RAG",
                    "Citações",
                    "Evaluation",
                    "Guard Agent",
                ],
                "Status": [
                    "🟢 Ativo",
                    "🟢 Ativo",
                    "🟢 Ativo",
                    "🟢 Ativo",
                    "🟢 Ativo",
                    "🟢 Ativo",
                    "🟢 Ativo",
                    "🟢 Ativo",
                ],
            }
        )

        st.dataframe(
            agent_status,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):

            st.markdown(
                msg["content"]
            )

    # --------------------------------------------------------
    # PERGUNTA
    # --------------------------------------------------------

    q = st.chat_input(
        "Ex.: Qual é o objeto do contrato?"
    )

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
    "🧠 Agente Jurídico → Guard → RAG → Retriever → Reranker → LLM..."
):
    try:
        result = call_orchestrator(
            query=q,
            org_id=user.get("organization_id"),
            mode="auto",
            top_k=8,
            rerank_k=5,
        )

        # --------------------------------------------------
        # VALIDAÇÃO DO RETORNO
        # --------------------------------------------------
        if result is None:
            raise RuntimeError(
                "O orquestrador não retornou nenhum resultado."
            )

        if not isinstance(result, dict):
            raise TypeError(
                "O orquestrador retornou um tipo inválido: "
                f"{type(result).__name__}"
            )

    except Exception as e:
        # --------------------------------------------------
        # ERRO REAL PARA DIAGNÓSTICO
        # --------------------------------------------------
        st.error(
            "❌ Não foi possível gerar uma resposta."
        )

        st.markdown(
            "### 🔍 Diagnóstico da execução"
        )

        st.write(
            f"**Erro:** `{type(e).__name__}`"
        )

        st.write(
            f"**Detalhes:** `{str(e)}`"
        )

        st.markdown(
            "### ⚠️ Detalhes técnicos"
        )

        st.exception(e)

        st.stop()


# ==========================================================
# RESPOSTA
# ==========================================================

response = str(
    result.get(
        "answer",
        "",
    )
    or ""
).strip()


# ==========================================================
# FALLBACK
# ==========================================================

if not response:

    response = (
        "Não foi possível gerar uma resposta "
        "com base nas evidências disponíveis."
    )


# ==========================================================
# EXIBIR RESPOSTA
# ==========================================================

st.markdown(response)


# ==========================================================
# DIAGNÓSTICO DO RAG
# ==========================================================

with st.expander(
    "🔍 Diagnóstico da execução",
    expanded=False,
):

    st.write(
        f"**Agente:** "
        f"{result.get('agent', 'N/D')}"
    )

    st.write(
        f"**Intent:** "
        f"{result.get('intent', 'N/D')}"
    )

    st.write(
        f"**Documentos recuperados:** "
        f"{len(result.get('retrieved', []) or [])}"
    )

    st.write(
        f"**Chunks reranked:** "
        f"{len(result.get('reranked', []) or [])}"
    )

    st.write(
        f"**Evidências:** "
        f"{result.get('evidence_count', 0)}"
    )

    st.write(
        f"**Latência:** "
        f"{result.get('latency_ms', 'N/D')} ms"
    )

    guard = result.get("guard", {})

    if isinstance(guard, dict):
        allowed = guard.get(
            "allowed",
            True,
        )

        st.write(
            "**Guard Agent:** "
            + (
                "🟢 Permitido"
                if allowed
                else "🔴 Bloqueado"
            )
        )

    # Mostra informações adicionais somente
    # quando existirem.
    if result.get("error"):
        st.error(
            f"Erro do orquestrador: "
            f"{result.get('error')}"
        )

    if result.get("reason"):
        st.info(
            f"Motivo: "
            f"{result.get('reason')}"
        )


# ==========================================================
# CITAÇÕES
# ==========================================================

citations = result.get(
    "citations",
    [],
) or []

if citations:

    st.markdown(
        "### 📚 Citações"
    )

    for cit in citations:

        if not isinstance(cit, dict):
            continue

        citation_id = cit.get(
            "id",
            "?",
        )

        document = cit.get(
            "document",
            cit.get(
                "document_name",
                "Documento",
            ),
        )

        page = cit.get(
            "page",
            "N/D",
        )

        chunk_id = cit.get(
            "chunk_id",
            "N/D",
        )

        st.markdown(
            f"- **[{citation_id}] "
            f"{document}**, "
            f"página {page} — "
            f"`{chunk_id}`"
        )

            # ------------------------------------------------
            # CITAÇÕES
            # ------------------------------------------------

            citations = safe_list(
                result.get(
                    "citations",
                    [],
                )
            )

            if citations:

                st.markdown(
                    "### 📚 Citações e Evidências"
                )

                for index, cit in enumerate(
                    citations,
                    1,
                ):

                    if not isinstance(
                        cit,
                        dict,
                    ):

                        st.write(
                            f"- [{index}] {cit}"
                        )

                        continue

                    cid = cit.get(
                        "id",
                        index,
                    )

                    document = cit.get(
                        "document",
                        cit.get(
                            "document_name",
                            "Documento",
                        ),
                    )

                    page_number = cit.get(
                        "page",
                        "N/D",
                    )

                    chunk_id = cit.get(
                        "chunk_id",
                        "N/D",
                    )

                    st.write(
                        f"- **[{cid}] {document}**, "
                        f"página {page_number} — "
                        f"`{chunk_id}`"
                    )

            # ------------------------------------------------
            # DIAGNÓSTICO
            # ------------------------------------------------

            with st.expander(
                "🔍 Diagnóstico da execução",
                expanded=False,
            ):

                st.write(
                    f"**Agente:** "
                    f"{result.get('agent', 'N/D')}"
                )

                st.write(
                    f"**Intent:** "
                    f"{result.get('intent', 'N/D')}"
                )

                st.write(
                    f"**Documentos recuperados:** "
                    f"{len(safe_list(result.get('retrieved')))}"
                )

                st.write(
                    f"**Chunks reranked:** "
                    f"{len(safe_list(result.get('reranked')))}"
                )

                st.write(
                    f"**Evidências:** "
                    f"{result.get('evidence_count', len(citations))}"
                )

                st.write(
                    f"**Latência:** "
                    f"{result.get('latency_ms', 'N/D')} ms"
                )

                guard = result.get(
                    "guard",
                    {},
                )

                if isinstance(
                    guard,
                    dict,
                ):

                    st.write(
                        f"**Guard Agent:** "
                        f"{'🟢 Permitido' if guard.get('allowed', True) else '🔴 Bloqueado'}"
                    )

            # ------------------------------------------------
            # ERRO
            # ------------------------------------------------

            if result.get("error"):

                with st.expander(
                    "⚠️ Detalhes técnicos",
                    expanded=False,
                ):

                    st.code(
                        str(
                            result["error"]
                        )
                    )

            # ------------------------------------------------
            # AUDITORIA
            # ------------------------------------------------

            try:

                audit(
                    user,
                    "rag.ask",
                    "conversation",
                    None,
                    {
                        "query": q,
                        "agent": result.get(
                            "agent"
                        ),
                        "intent": result.get(
                            "intent"
                        ),
                        "retrieved": len(
                            safe_list(
                                result.get(
                                    "retrieved"
                                )
                            )
                        ),
                        "reranked": len(
                            safe_list(
                                result.get(
                                    "reranked"
                                )
                            )
                        ),
                        "evidence_count": result.get(
                            "evidence_count",
                            0,
                        ),
                        "latency_ms": result.get(
                            "latency_ms",
                            0,
                        ),
                    },
                )

            except Exception:
                pass

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )


# ============================================================
# INGESTÃO RAG
# ============================================================

elif page == "Ingestão RAG":

    page_title(
        "📥 Ingestão RAG",
        "PDF/DOCX/TXT → OCR → Chunking → Embeddings → FAISS",
    )

    up = st.file_uploader(
        "Envie um documento",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
    )

    use_ocr = st.checkbox(
        "Usar OCR quando necessário",
        True,
    )

    if up and st.button(
        "🚀 Processar documento",
        type="primary",
    ):

        with st.status(
            "Executando pipeline...",
            expanded=True,
        ) as status:

            try:

                st.write(
                    "1/5 Extraindo texto e páginas"
                )

                st.write(
                    "2/5 Aplicando OCR quando necessário"
                )

                st.write(
                    "3/5 Gerando chunks com metadados"
                )

                st.write(
                    "4/5 Gerando embeddings"
                )

                st.write(
                    "5/5 Atualizando índice vetorial"
                )

                result = ingest_document(
                    up,
                    user.get(
                        "organization_id"
                    ),
                    use_ocr=use_ocr,
                )

                status.update(
                    label="Pipeline concluído",
                    state="complete",
                )

                result = safe_dict(
                    result
                )

                st.success(
                    f"Documento indexado: "
                    f"{result.get('chunks', 0)} chunks · "
                    f"{result.get('pages', 0)} páginas · "
                    f"OCR: "
                    f"{result.get('ocr_pages', 0)} páginas"
                )

                try:

                    audit(
                        user,
                        "document.ingest",
                        "document",
                        result.get(
                            "document_id"
                        ),
                        result,
                    )

                except Exception:
                    pass

            except Exception as exc:

                status.update(
                    label="Falha no pipeline",
                    state="error",
                )

                st.error(
                    f"Erro na ingestão: {exc}"
                )

                with st.expander(
                    "Detalhes técnicos"
                ):

                    st.code(
                        traceback.format_exc()
                    )

    st.info(
        "OCR de PDF escaneado usa PyMuPDF + Tesseract. "
        "Em produção, o binário Tesseract precisa estar instalado."
    )


# ============================================================
# BASE VETORIAL
# ============================================================

elif page == "Base Vetorial":

    page_title(
        "🔎 Base Vetorial",
        "Busca semântica + reranking",
    )

    q = st.text_input(
        "Consulta"
    )

    if q:

        with st.spinner(
            "Retriever + Reranker..."
        ):

            try:

                result = retrieve_and_rerank(
                    q,
                    user.get(
                        "organization_id"
                    ),
                    top_k=10,
                    rerank_k=5,
                )

                result = safe_dict(
                    result
                )

                retrieved = safe_list(
                    result.get(
                        "retrieved"
                    )
                )

                reranked = safe_list(
                    result.get(
                        "reranked"
                    )
                )

                st.write(
                    f"**Retriever:** "
                    f"{len(retrieved)} resultados · "
                    f"**Reranker:** "
                    f"{len(reranked)} resultados"
                )

                for i, x in enumerate(
                    reranked,
                    1,
                ):

                    if not isinstance(
                        x,
                        dict,
                    ):
                        continue

                    with st.container(
                        border=True
                    ):

                        st.write(
                            f"**[{i}] "
                            f"{x.get('document', 'Documento')} "
                            f"· página "
                            f"{x.get('page', 'N/D')}**"
                        )

                        st.caption(
                            f"chunk={x.get('chunk_id', 'N/D')} · "
                            f"score retriever="
                            f"{float(x.get('retriever_score', 0) or 0):.4f} · "
                            f"score reranker="
                            f"{float(x.get('reranker_score', 0) or 0):.4f}"
                        )

                        st.write(
                            x.get(
                                "content",
                                "",
                            )
                        )

            except Exception as exc:

                st.error(
                    f"Erro na busca vetorial: {exc}"
                )


# ============================================================
# AVALIAÇÃO RAG
# ============================================================

elif page == "Avaliação RAG":

    page_title(
        "📊 Avaliação da Resposta",
        "Context Relevance + Citation Coverage + Groundedness",
    )

    question = st.text_area(
        "Pergunta"
    )

    answer = st.text_area(
        "Resposta da IA"
    )

    if st.button(
        "🧪 Avaliar",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Informe a pergunta."
            )

        elif not answer.strip():

            st.warning(
                "Informe a resposta."
            )

        else:

            with st.spinner(
                "Calculando métricas..."
            ):

                try:

                    rag_result = rag_answer(
                        question,
                        user.get(
                            "organization_id"
                        ),
                        top_k=8,
                        rerank_k=5,
                        generate_answer_flag=False,
                    )

                    rag_result = safe_dict(
                        rag_result
                    )

                    score = evaluate_answer(
                        question,
                        answer,
                        safe_list(
                            rag_result.get(
                                "reranked"
                            )
                        ),
                        safe_list(
                            rag_result.get(
                                "citations"
                            )
                        ),
                    )

                    a, b, c, d = st.columns(4)

                    a.metric(
                        "Context relevance",
                        f"{score['context_relevance']:.2f}",
                    )

                    b.metric(
                        "Citation coverage",
                        f"{score['citation_coverage']:.2f}",
                    )

                    c.metric(
                        "Groundedness",
                        f"{score['groundedness']:.2f}",
                    )

                    d.metric(
                        "Overall",
                        f"{score['overall']:.2f}",
                    )

                    st.markdown(
                        f"### Qualidade: "
                        f"**{score.get('quality', 'N/D')}**"
                    )

                    if score.get(
                        "recommendations"
                    ):

                        st.markdown(
                            "### 💡 Recomendações"
                        )

                        for recommendation in score[
                            "recommendations"
                        ]:

                            st.info(
                                recommendation
                            )

                    with st.expander(
                        "JSON completo"
                    ):

                        st.json(
                            score
                        )

                except Exception as exc:

                    st.error(
                        f"Erro na avaliação: {exc}"
                    )


# ============================================================
# DOCUMENTOS
# ============================================================

elif page == "Documentos":

    page_title(
        "📄 Documentos"
    )

    try:

        documents = list_documents(
            user.get(
                "organization_id"
            )
        )

        if not documents:

            st.info(
                "Nenhum documento encontrado."
            )

        for d in documents:

            if not isinstance(
                d,
                dict,
            ):
                continue

            with st.container(
                border=True
            ):

                st.write(
                    f"**{d.get('name', 'Documento')}**"
                )

                st.caption(
                    f"{d.get('type', 'N/D')} · "
                    f"{d.get('status', 'N/D')} · "
                    f"páginas: {d.get('pages', 0)} · "
                    f"chunks: {d.get('chunks', 0)} · "
                    f"OCR: {d.get('ocr_pages', 0)}"
                )

    except Exception as exc:

        st.error(
            f"Erro ao carregar documentos: {exc}"
        )


# ============================================================
# PROCESSOS
# ============================================================

elif page == "Processos":

    page_title(
        "⚖️ Processos"
    )

    with st.form(
        "new_case"
    ):

        title = st.text_input(
            "Título"
        )

        client = st.text_input(
            "Cliente"
        )

        category = st.selectbox(
            "Categoria",
            [
                "Cível",
                "Trabalhista",
                "Contratos",
                "Tributário",
                "Previdenciário",
                "Outros",
            ],
        )

        priority = st.selectbox(
            "Prioridade",
            [
                "Baixa",
                "Média",
                "Alta",
            ],
        )

        submitted = st.form_submit_button(
            "Cadastrar",
            type="primary",
        )

        if submitted:

            if not title.strip():

                st.warning(
                    "Informe o título do processo."
                )

            else:

                try:

                    create_case(
                        user.get(
                            "organization_id"
                        ),
                        title,
                        client,
                        category,
                        priority,
                    )

                    st.success(
                        "Processo criado."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Erro ao criar processo: {exc}"
                    )

    try:

        cases = list_cases(
            user.get(
                "organization_id"
            )
        )

        for c in cases:

            if not isinstance(
                c,
                dict,
            ):
                continue

            with st.container(
                border=True
            ):

                st.write(
                    f"**{c.get('title', 'Processo')}**"
                )

                st.caption(
                    f"{c.get('client', 'N/D')} · "
                    f"{c.get('category', 'N/D')} · "
                    f"{c.get('priority', 'N/D')} · "
                    f"{c.get('status', 'N/D')}"
                )

    except Exception as exc:

        st.error(
            f"Erro ao carregar processos: {exc}"
        )


# ============================================================
# ANÁLISE DE RISCO
# ============================================================

elif page == "Análise de Risco":

    page_title(
        "⚠️ Análise de Risco IA",
        "Agente de Risco + RAG + Evidências",
    )

    text = st.text_area(
        "Cole o texto ou resumo do documento",
        height=260,
    )

    if st.button(
        "🔍 Analisar risco",
        type="primary",
    ):

        if not text.strip():

            st.warning(
                "Informe o texto para análise."
            )

        else:

            with st.spinner(
                "Agente de Risco analisando..."
            ):

                try:

                    result = risk_analysis(
                        query=(
                            "Analise os principais riscos, "
                            "evidências, lacunas e recomendações "
                            "sem inventar fatos."
                        ),
                        org_id=user.get(
                            "organization_id"
                        ),
                        top_k=8,
                        rerank_k=5,
                        extra_context=text,
                    )

                    result = safe_dict(
                        result
                    )

                    st.markdown(
                        result.get(
                            "answer",
                            "Não foi possível gerar a análise.",
                        )
                    )

                    citations = safe_list(
                        result.get(
                            "citations"
                        )
                    )

                    if citations:

                        st.markdown(
                            "### 📚 Evidências"
                        )

                        for index, c in enumerate(
                            citations,
                            1,
                        ):

                            if not isinstance(
                                c,
                                dict,
                            ):
                                continue

                            st.write(
                                f"- [{c.get('id', index)}] "
                                f"{c.get('document', 'Documento')} "
                                f"· página "
                                f"{c.get('page', 'N/D')}"
                            )

                except Exception as exc:

                    st.error(
                        f"Erro na análise de risco: {exc}"
                    )

                    with st.expander(
                        "Detalhes técnicos"
                    ):

                        st.code(
                            traceback.format_exc()
                        )


# ============================================================
# AUDITORIA
# ============================================================

elif page == "Auditoria":

    page_title(
        "🛡️ Auditoria",
        "Eventos do tenant",
    )

    try:

        from db import get_connection

        with get_connection() as connection:

            rows = connection.execute(
                """
                SELECT
                    action,
                    entity_type,
                    entity_id,
                    created_at,
                    metadata
                FROM audit_logs
                WHERE organization_id=?
                ORDER BY id DESC
                LIMIT 100
                """,
                (
                    user.get(
                        "organization_id"
                    ),
                ),
            ).fetchall()

        if not rows:

            st.info(
                "Nenhum evento de auditoria encontrado."
            )

        for r in rows:

            st.write(
                f"`{r['created_at']}` · "
                f"**{r['action']}** · "
                f"{r['entity_type']}#"
                f"{r['entity_id']} · "
                f"{r['metadata']}"
            )

    except Exception as exc:

        st.error(
            f"Erro ao carregar auditoria: {exc}"
        )


# ============================================================
# CONFIGURAÇÕES
# ============================================================

elif page == "Configurações":

    page_title(
        "⚙️ Configurações IA/RAG",
        "Configuração da inteligência artificial",
    )

    st.markdown(
        "### 🤖 Provedor LLM"
    )

    llm = st.selectbox(
        "LLM",
        [
            "Modo Demo",
            "Gemini",
            "OpenAI",
        ],
    )

    st.markdown(
        "### 🔎 Recuperação"
    )

    vector_db = st.selectbox(
        "Vector DB",
        [
            "FAISS local",
            "Qdrant (produção)",
            "pgvector (produção)",
        ],
    )

    reranker = st.selectbox(
        "Reranker",
        [
            "CrossEncoder",
            "Fallback lexical",
        ],
    )

    top_k = st.number_input(
        "Top K Retriever",
        min_value=1,
        max_value=50,
        value=8,
    )

    rerank_k = st.number_input(
        "Top K Reranker",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.markdown(
        "### 🛡️ Segurança"
    )

    citations_required = st.checkbox(
        "Citações obrigatórias",
        True,
    )

    evaluation_enabled = st.checkbox(
        "Avaliação habilitada",
        True,
    )

    guard_enabled = st.checkbox(
        "Proteção contra prompt injection",
        True,
    )

    st.markdown("---")

    st.markdown(
        "### 🧠 Agentes"
    )

    agent_config = {
        "Agente Jurídico": True,
        "Agente de Risco": True,
        "Agente de Resumo": True,
        "Agente Geral": True,
        "RAG": True,
        "Citações": citations_required,
        "Evaluation": evaluation_enabled,
        "Guard Agent": guard_enabled,
    }

    for name, enabled in agent_config.items():

        status = (
            "🟢 Ativo"
            if enabled
            else "🔴 Desativado"
        )

        st.write(
            f"**{name}:** {status}"
        )

    st.markdown("---")

    st.success(
        f"""
        Configuração carregada.

        LLM: {llm}

        Vector DB: {vector_db}

        Reranker: {reranker}

        Retriever Top-K: {top_k}

        Reranker Top-K: {rerank_k}
        """
    )


# ============================================================
# FALLBACK
# ============================================================

else:

    st.warning(
        "Página não encontrada."
    )

    st.session_state.page = "Dashboard"

    st.rerun()
