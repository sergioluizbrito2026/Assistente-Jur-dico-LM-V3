import os

RERANK_MODEL=os.getenv(
    "RERANK_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

def rerank(query, documents, top_k=5):
    if not documents:
        return []
    try:
        from sentence_transformers import CrossEncoder
        model=CrossEncoder(RERANK_MODEL)
        pairs=[(query,d["content"]) for d in documents]
        scores=model.predict(pairs)
        out=[]
        for d,s in zip(documents,scores):
            x=dict(d); x["reranker_score"]=float(s); out.append(x)
        return sorted(out,key=lambda x:x["reranker_score"],reverse=True)[:top_k]
    except Exception:
        # Fallback lexical reranker if the cross-encoder is unavailable.
        q=set(query.lower().split())
        out=[]
        for d in documents:
            words=set(d["content"].lower().split())
            x=dict(d); x["reranker_score"]=len(q&words)/max(1,len(q)); out.append(x)
        return sorted(out,key=lambda x:x["reranker_score"],reverse=True)[:top_k]
