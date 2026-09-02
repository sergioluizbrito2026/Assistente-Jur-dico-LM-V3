import streamlit as st
from pathlib import Path

from db import init_db, seed_demo
from services.auth import authenticate, get_current_user, logout
from services.ingestion import ingest_document
from services.rag_pipeline import rag_answer, retrieve_and_rerank
from services.evaluation import evaluate_answer
from services.documents import list_documents
from services.cases import list_cases, create_case
from services.audit import audit

st.set_page_config(
    page_title="Assistente Jurídico IA SaaS V3",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
seed_demo()

st.markdown("""
<style>
.stApp { background: #050d1a; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#09152b,#050d1a); }
.block-container { max-width: 1500px; }
[data-testid="stMetric"] { background:#0d1a30; border:1px solid #25385d; border-radius:14px; padding:14px; }
.v3 { padding:10px 14px; border:1px solid #2a3f68; border-radius:12px; background:#0b1830; }
</style>
""", unsafe_allow_html=True)

user = get_current_user()
if not user:
    st.markdown("<h1 style='text-align:center;margin-top:90px'>⚖️ Assistente Jurídico IA SaaS V3</h1>", unsafe_allow_html=True)
    st.caption("OCR → Chunking → Embeddings → Vector DB → Retriever → Reranker → LLM → Citações → Avaliação")
    with st.form("login"):
        email = st.text_input("E-mail", "admin@demo.local")
        password = st.text_input("Senha", "admin123", type="password")
        if st.form_submit_button("Entrar", type="primary", use_container_width=True):
            if authenticate(email, password):
                st.rerun()
            st.error("Credenciais inválidas.")
    st.info("Demo: admin@demo.local / admin123")
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("## ⚖️ Assistente Jurídico IA")
    st.caption("V3 · RAG Avançado")
    st.divider()

    menus = {
        "PRINCIPAL": ["Dashboard", "Assistente IA"],
        "INTELIGÊNCIA": ["Ingestão RAG", "Base Vetorial", "Avaliação RAG"],
        "GESTÃO JURÍDICA": ["Documentos", "Processos", "Análise de Risco"],
        "SISTEMA": ["Auditoria", "Configurações"],
    }
    for group, items in menus.items():
        st.caption(group)
        for item in items:
            if st.button(item, key="nav_"+item, use_container_width=True):
                st.session_state.page = item
                st.rerun()

    st.divider()
    st.caption(f"👤 {user['name']} · {user['role']}")
    if st.button("Sair", use_container_width=True):
        logout()
        st.rerun()

page = st.session_state.page

def page_title(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

if page == "Dashboard":
    page_title("Dashboard de Inteligência Jurídica", "Pipeline completo de IA Generativa + RAG")
    a,b,c,d = st.columns(4)
    with a: st.metric("📄 Documentos", len(list_documents(user["organization_id"])))
    with b: st.metric("⚖️ Processos", len(list_cases(user["organization_id"])))
    with c: st.metric("🧠 Pipeline", "9 etapas")
    with d: st.metric("🔎 RAG", "Ativo")
    st.markdown("### Arquitetura V3")
    stages = ["PDF/DOCX", "OCR", "Chunking", "Embeddings", "Vector DB", "Retriever", "Reranker", "LLM", "Citações", "Avaliação"]
    cols = st.columns(5)
    for i, stage in enumerate(stages):
        cols[i % 5].markdown(f'<div class="v3"><b>{i+1:02d}</b> · {stage}</div>', unsafe_allow_html=True)
    st.markdown("### Princípios")
    st.write("**Grounded answers:** a resposta deve ser fundamentada no contexto recuperado.")
    st.write("**Citações rastreáveis:** cada evidência aponta para documento e página quando disponíveis.")
    st.write("**Avaliação:** relevância do contexto, cobertura de citações e sinais de groundedness.")
    st.warning("Sistema de apoio. A saída da IA deve ser revisada por profissional habilitado.")

elif page == "Assistente IA":
    page_title("Assistente Jurídico IA", "LLM respondendo com RAG, reranking e citações")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    q = st.chat_input("Ex.: Quais cláusulas do contrato apresentam risco?")
    if q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.spinner("Executando Retriever → Reranker → LLM..."):
            result = rag_answer(q, user["organization_id"], top_k=8, rerank_k=5)
        response = result["answer"]
        if result["citations"]:
            response += "\n\n### 📚 Citações\n"
            for cit in result["citations"]:
                response += f"- **[{cit['id']}] {cit['document']}**, página {cit.get('page','N/D')} — `{cit['chunk_id']}`\n"
        st.session_state.messages.append({"role":"assistant","content":response})
        audit(user, "rag.ask", "conversation", None, {"query": q, "retrieved": len(result["retrieved"]), "reranked": len(result["reranked"])})
        st.rerun()

elif page == "Ingestão RAG":
    page_title("Ingestão RAG", "PDF/DOCX → OCR → chunking → embeddings → FAISS")
    up = st.file_uploader("Envie um documento", type=["pdf","docx","txt"])
    use_ocr = st.checkbox("Usar OCR quando necessário", True)
    if up and st.button("🚀 Processar documento", type="primary"):
        with st.status("Executando pipeline...", expanded=True) as status:
            st.write("1/5 Extraindo texto e páginas")
            st.write("2/5 Aplicando OCR quando necessário")
            st.write("3/5 Gerando chunks com metadados")
            st.write("4/5 Gerando embeddings")
            st.write("5/5 Atualizando índice vetorial")
            try:
                result = ingest_document(up, user["organization_id"], use_ocr=use_ocr)
                status.update(label="Pipeline concluído", state="complete")
                st.success(f"Documento indexado: {result['chunks']} chunks · {result['pages']} páginas · OCR: {result['ocr_pages']} páginas")
                audit(user, "document.ingest", "document", result["document_id"], result)
            except Exception as e:
                status.update(label="Falha no pipeline", state="error")
                st.error(str(e))
    st.info("OCR de PDF escaneado usa PyMuPDF + Tesseract. Em produção, instale o binário Tesseract no servidor.")

elif page == "Base Vetorial":
    page_title("Base Vetorial", "Busca semântica + reranking")
    q = st.text_input("Consulta")
    if q:
        with st.spinner("Retriever + Reranker..."):
            result = retrieve_and_rerank(q, user["organization_id"], top_k=10, rerank_k=5)
        st.write(f"**Retriever:** {len(result['retrieved'])} resultados · **Reranker:** {len(result['reranked'])} resultados")
        for i, x in enumerate(result["reranked"], 1):
            with st.container(border=True):
                st.write(f"**[{i}] {x['document']} · página {x.get('page','N/D')}**")
                st.caption(f"chunk={x['chunk_id']} · score retriever={x['retriever_score']:.4f} · score reranker={x['reranker_score']:.4f}")
                st.write(x["content"])

elif page == "Avaliação RAG":
    page_title("Avaliação da Resposta", "Métricas simples e rastreáveis para qualidade do RAG")
    question = st.text_area("Pergunta")
    answer = st.text_area("Resposta da IA")
    if st.button("Avaliar", type="primary") and question and answer:
        with st.spinner("Calculando métricas..."):
            result = rag_answer(question, user["organization_id"], top_k=8, rerank_k=5, generate_answer=False)
            score = evaluate_answer(question, answer, result["reranked"], result["citations"])
        a,b,c,d = st.columns(4)
        a.metric("Context relevance", f"{score['context_relevance']:.2f}")
        b.metric("Citation coverage", f"{score['citation_coverage']:.2f}")
        c.metric("Groundedness", f"{score['groundedness']:.2f}")
        d.metric("Overall", f"{score['overall']:.2f}")
        st.json(score)

elif page == "Documentos":
    page_title("Documentos")
    for d in list_documents(user["organization_id"]):
        with st.container(border=True):
            st.write(f"**{d['name']}**")
            st.caption(f"{d['type']} · {d['status']} · páginas: {d['pages']} · chunks: {d['chunks']} · OCR: {d['ocr_pages']}")

elif page == "Processos":
    page_title("Processos")
    with st.form("new_case"):
        title = st.text_input("Título")
        client = st.text_input("Cliente")
        category = st.selectbox("Categoria", ["Cível","Trabalhista","Contratos","Tributário","Previdenciário","Outros"])
        priority = st.selectbox("Prioridade", ["Baixa","Média","Alta"])
        if st.form_submit_button("Cadastrar", type="primary"):
            create_case(user["organization_id"], title, client, category, priority)
            st.success("Processo criado.")
            st.rerun()
    for c in list_cases(user["organization_id"]):
        with st.container(border=True):
            st.write(f"**{c['title']}**")
            st.caption(f"{c['client']} · {c['category']} · {c['priority']} · {c['status']}")

elif page == "Análise de Risco":
    page_title("Análise de Risco IA")
    text = st.text_area("Cole o texto ou resumo do documento", height=260)
    if st.button("Analisar risco", type="primary") and text:
        result = rag_answer(
            "Analise os principais riscos, evidências, lacunas e recomendações sem inventar fatos.",
            user["organization_id"], top_k=8, rerank_k=5, extra_context=text
        )
        st.markdown(result["answer"])
        if result["citations"]:
            st.markdown("### 📚 Evidências")
            for c in result["citations"]:
                st.write(f"- [{c['id']}] {c['document']} · página {c.get('page','N/D')}")

elif page == "Auditoria":
    page_title("Auditoria", "Eventos do tenant")
    from db import get_connection
    with get_connection() as c:
        rows = c.execute("SELECT action,entity_type,entity_id,created_at,metadata FROM audit_logs WHERE organization_id=? ORDER BY id DESC LIMIT 100", (user["organization_id"],)).fetchall()
    for r in rows:
        st.write(f"`{r['created_at']}` · **{r['action']}** · {r['entity_type']}#{r['entity_id']} · {r['metadata']}")

elif page == "Configurações":
    page_title("Configurações IA/RAG")
    st.selectbox("LLM", ["Modo Demo", "Gemini", "OpenAI"])
    st.selectbox("Vector DB", ["FAISS local", "Qdrant (produção)", "pgvector (produção)"])
    st.selectbox("Reranker", ["CrossEncoder", "Fallback lexical"])
    st.number_input("Top K Retriever", 1, 50, 8)
    st.number_input("Top K Reranker", 1, 20, 5)
    st.checkbox("Citações obrigatórias", True)
    st.checkbox("Avaliação habilitada", True)
    st.checkbox("Proteção contra prompt injection", True)
