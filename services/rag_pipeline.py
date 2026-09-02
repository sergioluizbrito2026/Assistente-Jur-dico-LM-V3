import os
from services.ai import generate_answer
from services.embeddings import semantic_search
from services.reranker import rerank


def retrieve_and_rerank(query, org_id, top_k=8, rerank_k=5):
  retrieved = semantic_search(query, org_id, top_k=top_k)
  reranked = rerank(query, retrieved, top_k=rerank_k)
  return {"retrieved": retrieved, "reranked": reranked}


def rag_answer(
    query,
    org_id,
    top_k=8,
    rerank_k=5,
    extra_context="",
    generate_answer_flag=True,
):
  result = retrieve_and_rerank(query, org_id, top_k, rerank_k)
  context = list(result["reranked"])
  if extra_context:
    context.insert(
        0,
        {
            "chunk_id": "user_input",
            "document": "Texto fornecido pelo usuário",
            "content": extra_context,
            "page": "N/D",
            "reranker_score": 1.0,
            "retriever_score": 1.0,
        },
    )
  citations = []
  for i, x in enumerate(context, 1):
    citations.append(
        {
            "id": i,
            "document": x["document"],
            "page": x.get("page", "N/D"),
            "chunk_id": x["chunk_id"],
        }
    )

  # Aqui usamos a função importada do services.ai e a flag renomeada
  answer = (
      generate_answer(query, context)
      if generate_answer_flag and callable(generate_answer)
      else ""
  )
  result["answer"] = answer
  result["citations"] = citations
  return result
