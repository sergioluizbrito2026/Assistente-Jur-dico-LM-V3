from pathlib import Path
from functools import lru_cache
from threading import RLock
import json
import os
import tempfile

import numpy as np

from db import get_connection


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = ROOT / "storage" / "vector"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Cache em memória dos índices FAISS
_INDEX_CACHE = {}

# Protege operações simultâneas no cache
_CACHE_LOCK = RLock()


# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================

@lru_cache(maxsize=1)
def _load_model():
    """
    Carrega o modelo de embeddings uma única vez por processo.

    Antes:
        O modelo era carregado novamente em cada pesquisa.

    Agora:
        O modelo permanece em memória.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


# ============================================================
# CAMINHOS DO ÍNDICE
# ============================================================

def _paths(org_id):
    """
    Retorna os arquivos FAISS e metadados da organização.
    """

    fp = INDEX_DIR / f"org_{org_id}.faiss"
    mp = INDEX_DIR / f"org_{org_id}.json"

    return fp, mp


def index_exists(org_id):
    """
    Verifica se o índice da organização existe.
    """

    fp, mp = _paths(org_id)

    return fp.exists() and mp.exists()


# ============================================================
# INVALIDAÇÃO DO CACHE
# ============================================================

def _invalidate_index_cache(org_id=None):
    """
    Remove índice(s) do cache em memória.
    """

    with _CACHE_LOCK:

        if org_id is None:
            _INDEX_CACHE.clear()
            return

        _INDEX_CACHE.pop(int(org_id), None)


# ============================================================
# CARREGAMENTO DO FAISS
# ============================================================

def _load_index(org_id):
    """
    Carrega FAISS + metadados.

    Utiliza cache em memória e verifica alteração dos arquivos.
    """

    import faiss

    org_id = int(org_id)

    fp, mp = _paths(org_id)

    if not fp.exists() or not mp.exists():
        return None, []

    try:
        faiss_mtime = fp.stat().st_mtime_ns
        meta_mtime = mp.stat().st_mtime_ns
    except OSError:
        return None, []

    with _CACHE_LOCK:

        cached = _INDEX_CACHE.get(org_id)

        if cached:

            cached_faiss_mtime = cached.get("faiss_mtime")
            cached_meta_mtime = cached.get("meta_mtime")

            if (
                cached_faiss_mtime == faiss_mtime
                and cached_meta_mtime == meta_mtime
            ):
                return cached["index"], cached["meta"]

        try:

            index = faiss.read_index(str(fp))

            meta = json.loads(
                mp.read_text(encoding="utf-8")
            )

            if not isinstance(meta, list):
                meta = []

            _INDEX_CACHE[org_id] = {
                "index": index,
                "meta": meta,
                "faiss_mtime": faiss_mtime,
                "meta_mtime": meta_mtime,
            }

            return index, meta

        except Exception:

            # Índice corrompido ou incompatível.
            _INDEX_CACHE.pop(org_id, None)

            return None, []


# ============================================================
# ESCRITA ATÔMICA
# ============================================================

def _atomic_write_index(org_id, index, meta):
    """
    Grava FAISS e JSON de forma atômica.

    Evita deixar o sistema com um índice parcialmente gravado
    caso o processo seja interrompido.
    """

    import faiss

    fp, mp = _paths(org_id)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    faiss_tmp = None
    meta_tmp = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".faiss",
            dir=INDEX_DIR,
            delete=False,
        ) as tmp:

            faiss_tmp = Path(tmp.name)

        faiss.write_index(
            index,
            str(faiss_tmp),
        )

        os.replace(
            faiss_tmp,
            fp,
        )

        # ----------------------------------------------------
        # METADADOS
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            dir=INDEX_DIR,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:

            meta_tmp = Path(tmp.name)

            json.dump(
                meta,
                tmp,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            tmp.flush()

        os.replace(
            meta_tmp,
            mp,
        )

    finally:

        if faiss_tmp and faiss_tmp.exists():
            try:
                faiss_tmp.unlink()
            except OSError:
                pass

        if meta_tmp and meta_tmp.exists():
            try:
                meta_tmp.unlink()
            except OSError:
                pass

    # Atualiza cache imediatamente
    _invalidate_index_cache(org_id)

    _load_index(org_id)


# ============================================================
# BUSCA DE CHUNKS NO BANCO
# ============================================================

def _fetch_chunks(org_id, document_id=None):
    """
    Busca chunks já trazendo o nome do documento através de JOIN.

    Isso elimina o problema N+1 que existia no semantic_search().
    """

    org_id = int(org_id)

    with get_connection() as c:

        if document_id is not None:

            rows = c.execute(
                """
                SELECT
                    ch.id,
                    ch.organization_id,
                    ch.content,
                    ch.document_id,
                    ch.page,
                    ch.chunk_index,
                    d.name AS document_name
                FROM chunks ch
                INNER JOIN documents d
                    ON d.id = ch.document_id
                WHERE
                    ch.organization_id = ?
                    AND ch.document_id = ?
                ORDER BY ch.id
                """,
                (
                    org_id,
                    int(document_id),
                ),
            ).fetchall()

        else:

            rows = c.execute(
                """
                SELECT
                    ch.id,
                    ch.organization_id,
                    ch.content,
                    ch.document_id,
                    ch.page,
                    ch.chunk_index,
                    d.name AS document_name
                FROM chunks ch
                INNER JOIN documents d
                    ON d.id = ch.document_id
                WHERE ch.organization_id = ?
                ORDER BY ch.id
                """,
                (org_id,),
            ).fetchall()

    return rows


# ============================================================
# CONVERSÃO DE METADADOS
# ============================================================

def _row_to_metadata(row):
    """
    Converte Row SQLite em metadata persistível.
    """

    return {
        "id": int(row["id"]),
        "organization_id": int(row["organization_id"]),
        "document_id": int(row["document_id"]),
        "document": row["document_name"] or "Desconhecido",
        "content": row["content"],
        "page": row["page"],
        "chunk_index": row["chunk_index"],
    }


# ============================================================
# CONSTRUÇÃO COMPLETA DO ÍNDICE
# ============================================================

def build_index_for_org(org_id):
    """
    Reconstrói completamente o índice de uma organização.

    Deve ser utilizado para:
        - primeira indexação;
        - recuperação;
        - manutenção;
        - reindexação manual.

    Não é chamado automaticamente a cada consulta.
    """

    org_id = int(org_id)

    rows = _fetch_chunks(org_id)

    if not rows:

        _invalidate_index_cache(org_id)

        return 0

    model = _load_model()

    texts = [
        row["content"]
        for row in rows
        if row["content"]
    ]

    if not texts:
        return 0

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=EMBED_BATCH_SIZE,
    )

    vectors = np.asarray(
        vectors,
        dtype="float32",
    )

    import faiss

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    meta = [
        _row_to_metadata(row)
        for row in rows
        if row["content"]
    ]

    _atomic_write_index(
        org_id,
        index,
        meta,
    )

    return len(meta)


# ============================================================
# INDEXAÇÃO INCREMENTAL
# ============================================================

def upsert_document_index(org_id, document_id):
    """
    Adiciona somente os chunks do documento ao índice existente.

    Antes:
        Rebuild completo da organização.

    Agora:
        Apenas os novos chunks são transformados em embeddings
        e adicionados ao FAISS.
    """

    org_id = int(org_id)
    document_id = int(document_id)

    rows = _fetch_chunks(
        org_id,
        document_id=document_id,
    )

    if not rows:
        return 0

    index, meta = _load_index(org_id)

    # --------------------------------------------------------
    # Se ainda não existe índice, cria um novo.
    # --------------------------------------------------------

    if index is None:

        return build_index_for_org(org_id)

    # --------------------------------------------------------
    # Identifica chunks que já foram indexados.
    # --------------------------------------------------------

    existing_ids = {
        int(item["id"])
        for item in meta
        if item.get("id") is not None
    }

    new_rows = [
        row
        for row in rows
        if int(row["id"]) not in existing_ids
    ]

    if not new_rows:
        return 0

    texts = [
        row["content"]
        for row in new_rows
        if row["content"]
    ]

    if not texts:
        return 0

    model = _load_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=EMBED_BATCH_SIZE,
    )

    vectors = np.asarray(
        vectors,
        dtype="float32",
    )

    # --------------------------------------------------------
    # Segurança dimensional
    # --------------------------------------------------------

    if index.d != vectors.shape[1]:

        # Índice incompatível com o modelo atual.
        # Rebuild completo é mais seguro.
        return build_index_for_org(org_id)

    # --------------------------------------------------------
    # Adiciona novos vetores
    # --------------------------------------------------------

    index.add(vectors)

    meta.extend(
        _row_to_metadata(row)
        for row in new_rows
        if row["content"]
    )

    _atomic_write_index(
        org_id,
        index,
        meta,
    )

    return len(new_rows)


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def semantic_search(query, org_id, top_k=10):
    """
    Busca semântica utilizando FAISS.

    O modelo e o índice ficam em memória sempre que possível.
    """

    org_id = int(org_id)

    query = (query or "").strip()

    if not query:
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 10

    top_k = max(1, min(top_k, 100))

    # --------------------------------------------------------
    # Carrega índice
    # --------------------------------------------------------

    index, meta = _load_index(org_id)

    if index is None:

        build_index_for_org(org_id)

        index, meta = _load_index(org_id)

    if index is None or not meta:

        return []

    if index.ntotal <= 0:

        return []

    # --------------------------------------------------------
    # Embedding da pergunta
    # --------------------------------------------------------

    model = _load_model()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32",
    )

    # --------------------------------------------------------
    # Pesquisa FAISS
    # --------------------------------------------------------

    search_k = min(
        top_k,
        index.ntotal,
    )

    scores, ids = index.search(
        query_vector,
        search_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0],
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(meta):
            continue

        item = meta[idx]

        # ----------------------------------------------------
        # Isolamento de organização
        # ----------------------------------------------------

        if int(
            item.get(
                "organization_id",
                -1,
            )
        ) != org_id:

            continue

        results.append(
            {
                "chunk_id": int(
                    item["id"]
                ),

                "document_id": int(
                    item["document_id"]
                ),

                "organization_id": org_id,

                "document": item.get(
                    "document",
                    "Desconhecido",
                ),

                "content": item.get(
                    "content",
                    "",
                ),

                "page": item.get(
                    "page",
                    "N/D",
                ),

                "chunk_index": item.get(
                    "chunk_index",
                    0,
                ),

                "retriever_score": float(
                    score
                ),
            }
        )

    return results


# ============================================================
# MANUTENÇÃO
# ============================================================

def rebuild_index(org_id):
    """
    Alias explícito para reindexação manual.
    """

    return build_index_for_org(org_id)


def clear_embedding_cache():
    """
    Limpa modelos e índices carregados em memória.

    Útil para:
        - troca de modelo;
        - manutenção;
        - testes.
    """

    _invalidate_index_cache()

    _load_model.cache_clear()
