import re

def _tokens(text):
    return set(re.findall(r"[a-zA-ZÀ-ÿ0-9]+", (text or "").lower()))

def context_relevance(question, chunks):
    q=_tokens(question)
    if not q or not chunks:
        return 0.0
    vals=[]
    for c in chunks:
        vals.append(len(q & _tokens(c["content"])) / max(1,len(q)))
    return min(1.0,sum(vals)/len(vals))

def citation_coverage(answer, citations):
    if not answer:
        return 0.0
    marks=set(re.findall(r"\[(\d+)\]",answer))
    valid={str(x["id"]) for x in citations}
    if not marks:
        return 0.0
    return len(marks & valid)/len(marks)

def groundedness(answer,chunks):
    # Heuristic baseline: proportion of answer content words appearing in retrieved context.
    # Replace with an LLM-as-judge/RAGAS evaluator in production.
    a=_tokens(answer)
    c=set()
    for x in chunks: c |= _tokens(x["content"])
    if not a: return 0.0
    return min(1.0,len(a&c)/len(a))

def evaluate_answer(question,answer,chunks,citations):
    cr=context_relevance(question,chunks)
    cc=citation_coverage(answer,citations)
    gr=groundedness(answer,chunks)
    overall=0.4*cr+0.3*cc+0.3*gr
    return {
        "context_relevance":round(cr,3),
        "citation_coverage":round(cc,3),
        "groundedness":round(gr,3),
        "overall":round(overall,3),
        "method":"heuristic baseline; production can use RAGAS/LLM-as-judge"
    }
