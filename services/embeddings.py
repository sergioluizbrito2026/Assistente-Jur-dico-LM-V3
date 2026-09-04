"""
Assistente Jurídico SaaS IA V3
services/embeddings.py

Serviço de embeddings e busca vetorial.

Responsabilidades:

    Documento/Chunks
          ↓
    SentenceTransformer
          ↓
    Vetores normalizados
          ↓
    FAISS
          ↓
    Busca semântica
          ↓
    Retriever
          ↓
    Reranker / RAG

Características:

- Isolamento por organização.
- Índice FAISS por organização.
- Metadados persistidos em JSON.
- Cache do modelo.
- Cache dos índices.
- Escrita atômica.
- Indexação completa.
- Indexação incremental.
- Rebuild automático quando necessário.
- Validação dimensional.
- Proteção contra índices corrompidos.
- Compatível com rag_pipeline.py V3.
- Não depende de Streamlit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Sequence
import json
import os
import tempfile

import numpy as np

from db import get_connection


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = (
    ROOT
    / "storage"
    / "vector"
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Modelo de embeddings
# ------------------------------------------------------------

EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
).strip()


# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------

try:
    EMBED_BATCH_SIZE = max(
        1,
        int(
            os.getenv(
                "EMBED_BATCH_SIZE",
                "32",
            )
        ),
    )
except (
    TypeError,
    ValueError,
):
    EMBED_BATCH_SIZE = 32


try:
    DEFAULT_TOP_K = max(
        1,
        int(
            os.getenv(
                "EMBEDDING_TOP_K",
                "10",
            )
        ),
    )
except (
    TypeError,
    ValueError,
):
    DEFAULT_TOP_K = 10


MAX_TOP_K = 100


# ============================================================
# CACHE
# ============================================================

_INDEX_CACHE: Dict[int, Dict[str, Any]] = {}

_CACHE_LOCK = RLock()


# ============================================================
# MODELO
# ============================================================

@lru_cache(maxsize=1)
def _load_model():
    """
    Carrega o modelo SentenceTransformer uma única vez
    por processo.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers não está instalado."
        ) from exc

    try:
        return SentenceTransformer(
            EMBED_MODEL
        )

    except Exception as exc:
        raise RuntimeError(
            f"Não foi possível carregar o modelo "
            f"de embeddings: {EMBED_MODEL}"
        ) from exc


def embedding_model_name() -> str:
    """
    Retorna o nome do modelo atualmente configurado.
    """

    return EMBED_MODEL


# ============================================================
# FAISS
# ============================================================

def _load_faiss():
    """
    Importa FAISS somente quando necessário.
    """

    try:
        import faiss

        return faiss

    except ImportError as exc:

        raise RuntimeError(
            "faiss-cpu não está instalado."
        ) from exc


# ============================================================
# CAMINHOS
# ============================================================

def _paths(
    org_id: int,
):
    """
    Retorna os caminhos do índice da organização.
    """

    org_id = int(org_id)

    faiss_path = (
        INDEX_DIR
        / f"org_{org_id}.faiss"
    )

    metadata_path = (
        INDEX_DIR
        / f"org_{org_id}.json"
    )

    return (
        faiss_path,
        metadata_path,
    )


def index_exists(
    org_id: int,
) -> bool:
    """
    Verifica se o índice vetorial existe.
    """

    faiss_path, metadata_path = _paths(
        org_id
    )

    return (
        faiss_path.is_file()
        and metadata_path.is_file()
    )


# ============================================================
# CACHE DOS ÍNDICES
# ============================================================

def _invalidate_index_cache(
    org_id: int | None = None,
):
    """
    Invalida o cache de um índice ou de todos os índices.
    """

    with _CACHE_LOCK:

        if org_id is None:

            _INDEX_CACHE.clear()

            return

        _INDEX_CACHE.pop(
            int(org_id),
            None,
        )


# ============================================================
# NORMALIZAÇÃO DE METADADOS
# ============================================================

def _normalize_metadata(
    meta: Any,
) -> List[Dict[str, Any]]:
    """
    Garante que os metadados tenham formato de lista.
    """

    if not isinstance(
        meta,
        list,
    ):
        return []

    normalized = []

    for item in meta:

        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get("id") is None:
            continue

        if item.get(
            "organization_id"
        ) is None:
            continue

        if item.get(
            "document_id"
        ) is None:
            continue

        if not item.get(
            "content"
        ):
            continue

        try:

            item = dict(item)

            item["id"] = int(
                item["id"]
            )

            item[
                "organization_id"
            ] = int(
                item[
                    "organization_id"
                ]
            )

            item[
                "document_id"
            ] = int(
                item[
                    "document_id"
                ]
            )

            normalized.append(
                item
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    return normalized


# ============================================================
# CARREGAMENTO DO ÍNDICE
# ============================================================

def _load_index(
    org_id: int,
):
    """
    Carrega FAISS e metadados.

    Utiliza cache em memória e verifica se os arquivos
    foram modificados.
    """

    faiss = _load_faiss()

    org_id = int(
        org_id
    )

    faiss_path, metadata_path = _paths(
        org_id
    )

    if not (
        faiss_path.exists()
        and metadata_path.exists()
    ):
        return (
            None,
            [],
        )

    try:

        faiss_mtime = (
            faiss_path.stat()
            .st_mtime_ns
        )

        metadata_mtime = (
            metadata_path.stat()
            .st_mtime_ns
        )

    except OSError:

        return (
            None,
            [],
        )

    with _CACHE_LOCK:

        cached = _INDEX_CACHE.get(
            org_id
        )

        if cached:

            if (
                cached.get(
                    "faiss_mtime"
                )
                == faiss_mtime
                and cached.get(
                    "metadata_mtime"
                )
                == metadata_mtime
            ):

                return (
                    cached.get(
                        "index"
                    ),
                    cached.get(
                        "meta",
                        [],
                    ),
                )

        try:

            index = faiss.read_index(
                str(
                    faiss_path
                )
            )

            raw_metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            meta = _normalize_metadata(
                raw_metadata
            )

            # ------------------------------------------------
            # Validação básica
            # ------------------------------------------------

            if index.ntotal != len(meta):

                # Índice e metadados estão inconsistentes.
                _INDEX_CACHE.pop(
                    org_id,
                    None,
                )

                return (
                    None,
                    [],
                )

            _INDEX_CACHE[
                org_id
            ] = {
                "index": index,
                "meta": meta,
                "faiss_mtime": faiss_mtime,
                "metadata_mtime": metadata_mtime,
            }

            return (
                index,
                meta,
            )

        except Exception:

            _INDEX_CACHE.pop(
                org_id,
                None,
            )

            return (
                None,
                [],
            )


# ============================================================
# ESCRITA ATÔMICA
# ============================================================

def _atomic_write_index(
    org_id: int,
    index: Any,
    meta: Sequence[Dict[str, Any]],
):
    """
    Salva FAISS e metadados de maneira atômica.
    """

    faiss = _load_faiss()

    org_id = int(
        org_id
    )

    faiss_path, metadata_path = _paths(
        org_id
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss_tmp = None
    metadata_tmp = None

    try:

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".faiss",
            prefix=f"org_{org_id}_",
            dir=INDEX_DIR,
            delete=False,
        ) as tmp:

            faiss_tmp = Path(
                tmp.name
            )

        faiss.write_index(
            index,
            str(
                faiss_tmp
            ),
        )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix=f"org_{org_id}_",
            dir=INDEX_DIR,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:

            metadata_tmp = Path(
                tmp.name
            )

            json.dump(
                list(meta),
                tmp,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

            tmp.flush()

            os.fsync(
                tmp.fileno()
            )

        # ----------------------------------------------------
        # Substituição
        # ----------------------------------------------------

        os.replace(
            faiss_tmp,
            faiss_path,
        )

        os.replace(
            metadata_tmp,
            metadata_path,
        )

    finally:

        if (
            faiss_tmp is not None
            and faiss_tmp.exists()
        ):

            try:
                faiss_tmp.unlink()
            except OSError:
                pass

        if (
            metadata_tmp is not None
            and metadata_tmp.exists()
        ):

            try:
                metadata_tmp.unlink()
            except OSError:
                pass

    _invalidate_index_cache(
        org_id
    )

    _load_index(
        org_id
    )


# ============================================================
# BANCO DE DADOS
# ============================================================

def _fetch_chunks(
    org_id: int,
    document_id: int | None = None,
):
    """
    Recupera chunks da organização.

    O nome do documento é obtido através de JOIN.
    """

    org_id = int(
        org_id
    )

    with get_connection() as connection:

        if document_id is not None:

            rows = connection.execute(
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

            rows = connection.execute(
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
                ORDER BY ch.id
                """,
                (
                    org_id,
                ),
            ).fetchall()

    return rows


def _row_to_metadata(
    row,
) -> Dict[str, Any]:
    """
    Converte uma linha SQLite para metadata do índice.
    """

    content = (
        row["content"]
        or ""
    ).strip()

    return {
        "id": int(
            row["id"]
        ),
        "organization_id": int(
            row["organization_id"]
        ),
        "document_id": int(
            row["document_id"]
        ),
        "document": (
            row["document_name"]
            or "Desconhecido"
        ),
        "content": content,
        "page": row["page"],
        "chunk_index": row[
            "chunk_index"
        ],
    }


# ============================================================
# ENCODING
# ============================================================

def _encode_texts(
    texts: Sequence[str],
) -> np.ndarray:
    """
    Gera embeddings normalizados.
    """

    if not texts:

        return np.empty(
            (
                0,
                0,
            ),
            dtype="float32",
        )

    model = _load_model()

    try:

        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBED_BATCH_SIZE,
        )

    except Exception as exc:

        raise RuntimeError(
            "Falha ao gerar embeddings."
        ) from exc

    vectors = np.asarray(
        vectors,
        dtype="float32",
    )

    if vectors.ndim != 2:

        raise RuntimeError(
            "O modelo de embeddings retornou "
            "um formato inválido."
        )

    if vectors.shape[0] != len(texts):

        raise RuntimeError(
            "Quantidade de embeddings diferente "
            "da quantidade de textos."
        )

    return vectors


# ============================================================
# CONSTRUÇÃO DO ÍNDICE
# ============================================================

def build_index_for_org(
    org_id: int,
) -> int:
    """
    Reconstrói completamente o índice de uma organização.
    """

    org_id = int(
        org_id
    )

    rows = _fetch_chunks(
        org_id
    )

    valid_rows = [
        row
        for row in rows
        if (
            row["content"]
            and str(
                row["content"]
            ).strip()
        )
    ]

    # --------------------------------------------------------
    # Organização sem documentos
    # --------------------------------------------------------

    if not valid_rows:

        _invalidate_index_cache(
            org_id
        )

        faiss_path, metadata_path = _paths(
            org_id
        )

        for path in (
            faiss_path,
            metadata_path,
        ):

            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

        return 0

    texts = [
        str(
            row["content"]
        ).strip()
        for row in valid_rows
    ]

    vectors = _encode_texts(
        texts
    )

    faiss = _load_faiss()

    dimension = int(
        vectors.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        vectors
    )

    meta = [
        _row_to_metadata(
            row
        )
        for row in valid_rows
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

def upsert_document_index(
    org_id: int,
    document_id: int,
) -> int:
    """
    Adiciona ao índice somente os chunks ainda não indexados
    de um documento.

    Se o índice estiver ausente ou inconsistente,
    reconstrói a organização.
    """

    org_id = int(
        org_id
    )

    document_id = int(
        document_id
    )

    rows = _fetch_chunks(
        org_id,
        document_id=document_id,
    )

    rows = [
        row
        for row in rows
        if (
            row["content"]
            and str(
                row["content"]
            ).strip()
        )
    ]

    if not rows:
        return 0

    index, meta = _load_index(
        org_id
    )

    # --------------------------------------------------------
    # Índice inexistente
    # --------------------------------------------------------

    if index is None:

        return build_index_for_org(
            org_id
        )

    # --------------------------------------------------------
    # IDs existentes
    # --------------------------------------------------------

    existing_ids = {
        int(
            item["id"]
        )
        for item in meta
        if item.get("id") is not None
    }

    new_rows = [
        row
        for row in rows
        if int(
            row["id"]
        )
        not in existing_ids
    ]

    if not new_rows:
        return 0

    texts = [
        str(
            row["content"]
        ).strip()
        for row in new_rows
    ]

    vectors = _encode_texts(
        texts
    )

    # --------------------------------------------------------
    # Verificação dimensional
    # --------------------------------------------------------

    if (
        index.d
        != vectors.shape[1]
    ):

        return build_index_for_org(
            org_id
        )

    # --------------------------------------------------------
    # Adição
    # --------------------------------------------------------

    index.add(
        vectors
    )

    meta.extend(
        _row_to_metadata(
            row
        )
        for row in new_rows
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

def semantic_search(
    query: str,
    org_id: int,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Executa busca semântica utilizando FAISS.

    Retorna uma lista compatível com:

        services.rag_pipeline
        services.reranker
        services.ai
    """

    org_id = int(
        org_id
    )

    query = (
        query
        or ""
    ).strip()

    if not query:
        return []

    # --------------------------------------------------------
    # top_k
    # --------------------------------------------------------

    try:

        top_k = int(
            top_k
        )

    except (
        TypeError,
        ValueError,
    ):

        top_k = DEFAULT_TOP_K

    top_k = max(
        1,
        min(
            top_k,
            MAX_TOP_K,
        ),
    )

    # --------------------------------------------------------
    # Índice
    # --------------------------------------------------------

    index, meta = _load_index(
        org_id
    )

    if (
        index is None
        or not meta
    ):

        try:

            build_index_for_org(
                org_id
            )

        except Exception:
            return []

        index, meta = _load_index(
            org_id
        )

    if (
        index is None
        or not meta
        or index.ntotal <= 0
    ):
        return []

    # --------------------------------------------------------
    # Segurança
    # --------------------------------------------------------

    if index.ntotal != len(meta):

        try:

            build_index_for_org(
                org_id
            )

            index, meta = _load_index(
                org_id
            )

        except Exception:
            return []

    if (
        index is None
        or not meta
    ):
        return []

    # --------------------------------------------------------
    # Embedding da consulta
    # --------------------------------------------------------

    try:

        query_vector = _encode_texts(
            [query]
        )

    except Exception:

        return []

    # --------------------------------------------------------
    # Busca FAISS
    # --------------------------------------------------------

    search_k = min(
        top_k,
        index.ntotal,
    )

    try:

        scores, ids = index.search(
            query_vector,
            search_k,
        )

    except Exception:

        return []

    results = []

    for score, position in zip(
        scores[0],
        ids[0],
    ):

        position = int(
            position
        )

        if (
            position < 0
            or position >= len(meta)
        ):
            continue

        item = meta[
            position
        ]

        # ----------------------------------------------------
        # Isolamento da organização
        # ----------------------------------------------------

        try:

            item_org_id = int(
                item.get(
                    "organization_id",
                    -1,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if item_org_id != org_id:
            continue

        # ----------------------------------------------------
        # Conteúdo
        # ----------------------------------------------------

        content = (
            item.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            continue

        # ----------------------------------------------------
        # Resultado padronizado
        # ----------------------------------------------------

        results.append(
            {
                "chunk_id": int(
                    item["id"]
                ),

                "document_id": int(
                    item[
                        "document_id"
                    ]
                ),

                "organization_id": org_id,

                "document": item.get(
                    "document",
                    "Desconhecido",
                ),

                "content": content,

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
# BUSCA COM FILTRO POR DOCUMENTO
# ============================================================

def semantic_search_document(
    query: str,
    org_id: int,
    document_id: int,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Busca semântica restrita a um documento.

    Utiliza os chunks do documento diretamente e calcula
    embeddings temporários para a busca.
    """

    org_id = int(
        org_id
    )

    document_id = int(
        document_id
    )

    query = (
        query
        or ""
    ).strip()

    if not query:
        return []

    rows = _fetch_chunks(
        org_id,
        document_id=document_id,
    )

    rows = [
        row
        for row in rows
        if (
            row["content"]
            and str(
                row["content"]
            ).strip()
        )
    ]

    if not rows:
        return []

    try:

        top_k = int(
            top_k
        )

    except (
        TypeError,
        ValueError,
    ):

        top_k = DEFAULT_TOP_K

    top_k = max(
        1,
        min(
            top_k,
            MAX_TOP_K,
        ),
    )

    texts = [
        str(
            row["content"]
        ).strip()
        for row in rows
    ]

    try:

        vectors = _encode_texts(
            texts
        )

        query_vector = _encode_texts(
            [query]
        )

    except Exception:

        return []

    # --------------------------------------------------------
    # Similaridade por produto escalar.
    #
    # Como os embeddings estão normalizados,
    # equivale à similaridade de cosseno.
    # --------------------------------------------------------

    scores = np.dot(
        vectors,
        query_vector[0],
    )

    ranked_indices = np.argsort(
        -scores
    )[
        : min(
            top_k,
            len(scores),
        )
    ]

    results = []

    for position in ranked_indices:

        row = rows[
            int(position)
        ]

        results.append(
            {
                "chunk_id": int(
                    row["id"]
                ),

                "document_id": int(
                    row["document_id"]
                ),

                "organization_id": org_id,

                "document": (
                    row[
                        "document_name"
                    ]
                    or "Desconhecido"
                ),

                "content": str(
                    row["content"]
                ).strip(),

                "page": row["page"],

                "chunk_index": row[
                    "chunk_index"
                ],

                "retriever_score": float(
                    scores[
                        int(position)
                    ]
                ),
            }
        )

    return results


# ============================================================
# MANUTENÇÃO
# ============================================================

def rebuild_index(
    org_id: int,
) -> int:
    """
    Rebuild explícito do índice.
    """

    return build_index_for_org(
        org_id
    )


def delete_index(
    org_id: int,
) -> bool:
    """
    Remove o índice vetorial de uma organização.
    """

    org_id = int(
        org_id
    )

    faiss_path, metadata_path = _paths(
        org_id
    )

    removed = False

    _invalidate_index_cache(
        org_id
    )

    for path in (
        faiss_path,
        metadata_path,
    ):

        try:

            if path.exists():

                path.unlink()

                removed = True

        except OSError:
            pass

    return removed


def clear_embedding_cache():
    """
    Limpa:

    - modelo SentenceTransformer;
    - índices FAISS em memória.
    """

    _invalidate_index_cache()

    _load_model.cache_clear()


# ============================================================
# ESTATÍSTICAS DO ÍNDICE
# ============================================================

def index_stats(
    org_id: int,
) -> Dict[str, Any]:
    """
    Retorna informações básicas do índice.
    """

    org_id = int(
        org_id
    )

    index, meta = _load_index(
        org_id
    )

    if (
        index is None
        or not meta
    ):

        return {
            "organization_id": org_id,
            "exists": False,
            "vectors": 0,
            "metadata": 0,
            "dimension": 0,
            "model": EMBED_MODEL,
        }

    return {
        "organization_id": org_id,
        "exists": True,
        "vectors": int(
            index.ntotal
        ),
        "metadata": len(
            meta
        ),
        "dimension": int(
            index.d
        ),
        "model": EMBED_MODEL,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

def embeddings_status() -> Dict[str, Any]:
    """
    Retorna o estado do serviço de embeddings.

    Não carrega o modelo para evitar custo desnecessário.
    """

    try:

        import sentence_transformers

        sentence_transformers_version = (
            getattr(
                sentence_transformers,
                "__version__",
                "unknown",
            )
        )

    except Exception:

        sentence_transformers_version = None

    try:

        import faiss

        faiss_available = True

        faiss_version = getattr(
            faiss,
            "__version__",
            "unknown",
        )

    except Exception:

        faiss_available = False
        faiss_version = None

    return {
        "status": (
            "ready"
            if faiss_available
            and sentence_transformers_version
            else "not_ready"
        ),

        "model": EMBED_MODEL,

        "batch_size": EMBED_BATCH_SIZE,

        "faiss_available": faiss_available,

        "faiss_version": faiss_version,

        "sentence_transformers_version": (
            sentence_transformers_version
        ),

        "index_directory": str(
            INDEX_DIR
        ),
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural.

    Não carrega modelo nem cria índice.
    """

    required = [
        "_load_model",
        "_load_index",
        "_fetch_chunks",
        "_encode_texts",
        "build_index_for_org",
        "upsert_document_index",
        "semantic_search",
        "semantic_search_document",
        "rebuild_index",
        "delete_index",
        "index_exists",
        "index_stats",
        "embeddings_status",
        "clear_embedding_cache",
    ]

    missing = [
        name
        for name in required
        if name not in globals()
    ]

    return {
        "valid": not missing,

        "module": (
            "services.embeddings"
        ),

        "required_functions": required,

        "missing_functions": missing,

        "model": EMBED_MODEL,

        "batch_size": EMBED_BATCH_SIZE,

        "index_directory": str(
            INDEX_DIR
        ),
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 70)
    print(
        "EMBEDDINGS.PY V3 - SELF TEST"
    )
    print("=" * 70)

    print(
        "Status              : "
        f"{'OK' if result['valid'] else 'ERRO'}"
    )

    print(
        "Modelo              : "
        f"{result['model']}"
    )

    print(
        "Batch size          : "
        f"{result['batch_size']}"
    )

    print(
        "Funções obrigatórias: "
        f"{len(result['required_functions'])}"
    )

    print(
        "Funções ausentes    : "
        f"{result['missing_functions']}"
    )

    print(
        "Diretório dos índices:"
    )

    print(
        f"  {result['index_directory']}"
    )

    print("=" * 70)
