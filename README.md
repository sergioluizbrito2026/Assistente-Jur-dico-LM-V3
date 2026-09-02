# ⚖️ Assistente Jurídico IA SaaS — V3

## Projeto focado em IA Generativa + RAG avançado

Esta versão implementa o fluxo:

**PDF/DOCX → OCR → Chunking → Embeddings → Vector Database → Retriever → Reranker → LLM → Citações → Avaliação**

A aplicação é um protótipo SaaS para apoio à análise e organização de informações jurídicas.

---

## 🧠 Pipeline

```text
                 ┌──────────────────────┐
                 │       PDF / DOCX     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Extração + OCR       │
                 │ PyMuPDF + Tesseract  │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Chunking + Metadata   │
                 │ página / documento    │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Embeddings            │
                 │ Sentence Transformers│
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Vector Database       │
                 │ FAISS (V3)             │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Retriever             │
                 │ busca semântica       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Reranker              │
                 │ CrossEncoder          │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ LLM                   │
                 │ Gemini / OpenAI       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Citações              │
                 │ documento + página    │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Avaliação             │
                 │ relevance / grounded  │
                 └──────────────────────┘
```

## ✨ Funcionalidades

- Dashboard SaaS
- Login com senha hash
- Multi-tenant por `organization_id`
- Upload PDF/DOCX/TXT
- Extração por página
- OCR para PDF escaneado
- Chunking com overlap
- Metadados de documento/página/chunk
- Embeddings multilíngues
- FAISS como vector database local
- Busca semântica
- CrossEncoder reranker
- Fallback lexical
- Gemini
- OpenAI
- Modo demo
- Respostas fundamentadas
- Citações rastreáveis
- Avaliação da resposta
- Métricas de context relevance
- Citation coverage
- Groundedness
- Score geral
- Auditoria
- Processos
- Análise de risco
- Configuração do pipeline

---

## 🚀 Instalação

### Windows

É necessário instalar o Tesseract OCR para PDFs escaneados.

Depois:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Docker

O arquivo `docker/tesseract.Dockerfile` instala Tesseract + idiomas português/inglês.

```bash
docker build -f docker/tesseract.Dockerfile -t assistente-juridico-v3 .
docker run -p 8501:8501 assistente-juridico-v3
```

---

## 🔑 Ativar Gemini

Defina:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave
```

Ou use Secrets no ambiente de deploy.

## 🔑 Ativar OpenAI

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=sua_chave
```

**Nunca publique API keys no GitHub.**

---

# 📚 Como funciona o RAG

### 1. Ingestão

O usuário envia o documento.

### 2. Extração

PDF com texto → PyMuPDF.

PDF escaneado → renderização → Tesseract OCR.

DOCX → python-docx.

### 3. Chunking

O texto é dividido em blocos com overlap para preservar contexto.

Cada chunk guarda:

- documento;
- página;
- índice;
- conteúdo;
- estimativa de tokens;
- metadados.

### 4. Embeddings

Cada chunk é transformado em vetor usando Sentence Transformers.

Modelo padrão:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

### 5. Vector DB

Na V3 o armazenamento local usa:

```text
FAISS IndexFlatIP
```

A arquitetura permite substituir por:

```text
Qdrant
pgvector
Pinecone
Weaviate
```

### 6. Retriever

A pergunta também vira embedding e o FAISS recupera os chunks mais semelhantes.

### 7. Reranker

Os candidatos passam por CrossEncoder para melhorar a ordem de relevância.

### 8. LLM

Somente o contexto recuperado é enviado para o modelo.

### 9. Citações

A resposta recebe referências:

```text
[1] Contrato.pdf — página 4
[2] Contrato.pdf — página 7
```

### 10. Avaliação

A V3 calcula uma linha de base de:

```text
Context Relevance
Citation Coverage
Groundedness
Overall
```

Para produção, substituir/expandir com RAGAS e LLM-as-a-judge.

---

# 🧪 Avaliação profissional

A evolução recomendada é montar um dataset com:

```json
{"question":"Qual é o prazo de vigência?","expected_sources":["Contrato.pdf"],"answer":"..."}
```

E acompanhar:

- Recall@K
- Precision@K
- MRR
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevance
- Citation Accuracy
- Latência
- custo por consulta

Isso transforma o projeto de "chatbot com documentos" em um projeto demonstrável de **Engenharia de IA/RAG**.

---

# 🏢 Próxima arquitetura de produção

A V3 usa FAISS local para ser fácil de executar.

Para SaaS real:

```text
Streamlit / Frontend
        ↓
FastAPI
        ↓
Auth + RBAC + Tenant
        ↓
Celery / Redis
        ↓
Document Pipeline
        ↓
Object Storage
        ↓
PostgreSQL + pgvector
        ↓
RAG Service
        ├── Retriever
        ├── Reranker
        └── Citation Engine
        ↓
LLM Gateway
        ├── Gemini
        └── OpenAI
        ↓
Evaluation / Observability
```

---

## ⚠️ Segurança

Para produção, implementar obrigatoriamente:

- isolamento por tenant;
- autorização por recurso;
- criptografia em repouso e trânsito;
- gestão segura de secrets;
- logs de auditoria;
- rate limiting;
- proteção contra prompt injection;
- sanitização de documentos;
- limites de upload;
- antivírus para arquivos;
- política de retenção;
- controle de custos;
- mascaramento de dados sensíveis.

## ⚖️ Uso jurídico

O sistema é uma ferramenta tecnológica de apoio. A saída da IA não substitui análise, decisão ou responsabilidade de profissional habilitado.

---

## 🎯 Objetivo de portfólio

Este projeto demonstra competências em:

**Python · Streamlit · LLM · RAG · OCR · Embeddings · Vector Search · FAISS · CrossEncoder · Prompt Engineering · Avaliação de IA · SQL · Multi-tenancy · Segurança · Automação**

