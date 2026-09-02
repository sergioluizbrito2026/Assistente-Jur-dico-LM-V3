from pathlib import Path
import json, os
import numpy as np
from db import get_connection

ROOT=Path(__file__).resolve().parents[1]
INDEX_DIR=ROOT/"storage"/"vector"
INDEX_DIR.mkdir(parents=True,exist_ok=True)

EMBED_MODEL=os.getenv("EMBEDDING_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

def _paths(org_id):
    return INDEX_DIR/f"org_{org_id}.faiss", INDEX_DIR/f"org_{org_id}.json"

def build_index_for_org(org_id):
    with get_connection() as c:
        rows=c.execute(
            "SELECT id,content,document_id,page,chunk_index FROM chunks WHERE organization_id=? ORDER BY id",
            (org_id,)
        ).fetchall()
    if not rows:
        return 0
    model=_load_model()
    texts=[r["content"] for r in rows]
    vectors=model.encode(texts,normalize_embeddings=True,show_progress_bar=False)
    vectors=np.asarray(vectors,dtype="float32")
    import faiss
    index=faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    fp,mp=_paths(org_id)
    faiss.write_index(index,str(fp))
    meta=[dict(r) for r in rows]
    mp.write_text(json.dumps(meta,ensure_ascii=False),encoding="utf-8")
    return len(rows)

def upsert_document_index(org_id, document_id):
    # Rebuild is intentionally simple for V3 prototype.
    # Production should use incremental vector upserts (Qdrant/pgvector).
    return build_index_for_org(org_id)

def semantic_search(query,org_id,top_k=10):
    fp,mp=_paths(org_id)
    if not fp.exists() or not mp.exists():
        build_index_for_org(org_id)
    import faiss
    model=_load_model()
    q=model.encode([query],normalize_embeddings=True)
    q=np.asarray(q,dtype="float32")
    index=faiss.read_index(str(fp))
    scores,ids=index.search(q,top_k)
    meta=json.loads(mp.read_text(encoding="utf-8"))
    results=[]
    for score,idx in zip(scores[0],ids[0]):
        if idx<0 or idx>=len(meta): continue
        m=meta[idx]
        with get_connection() as c:
            row=c.execute("SELECT name FROM documents WHERE id=?",(m["document_id"],)).fetchone()
        results.append({
            "chunk_id":m["id"],"document":row["name"] if row else "Desconhecido",
            "content":m["content"],"page":m["page"],"chunk_index":m["chunk_index"],
            "retriever_score":float(score)
        })
    return results
