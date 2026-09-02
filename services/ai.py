import os

SYSTEM_PROMPT = """
Você é um Assistente Jurídico de apoio à análise documental.
Regras:
1. Responda somente com base no contexto fornecido quando a pergunta depender de documentos.
2. Não invente fatos, artigos de lei, jurisprudência, precedentes ou números de processo.
3. Se o contexto for insuficiente, diga explicitamente que não há evidência suficiente.
4. Diferencie fato encontrado no documento de interpretação/recomendação.
5. Use as marcas [1], [2], etc. para indicar as fontes do contexto.
6. Não trate a resposta como decisão jurídica definitiva.
7. Seja objetivo, estruturado e profissional.
"""

def _context(chunks):
    return "\n\n".join(
        f"[{i}] DOCUMENTO: {x['document']} | PÁGINA: {x.get('page','N/D')} | CHUNK: {x['chunk_id']}\n{x['content']}"
        for i,x in enumerate(chunks,1)
    )

def generate_answer(query,chunks):
    prompt=SYSTEM_PROMPT+"\n\nCONTEXTO:\n"+_context(chunks)+"\n\nPERGUNTA:\n"+query
    provider=os.getenv("LLM_PROVIDER","demo").lower()

    if provider=="gemini" and os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai
            client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            model=os.getenv("GEMINI_MODEL","gemini-2.5-flash")
            response=client.models.generate_content(model=model,contents=prompt)
            return response.text
        except Exception as e:
            return f"Falha no Gemini: {e}"

    if provider=="openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model=os.getenv("OPENAI_MODEL","gpt-5-mini")
            response=client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":"CONTEXTO:\n"+_context(chunks)+"\n\nPERGUNTA:\n"+query}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Falha no OpenAI: {e}"

    return """### 🤖 Modo demonstração

O pipeline **Retriever → Reranker → LLM** foi executado, mas nenhuma API de LLM está configurada.

Configure `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` ou `LLM_PROVIDER=openai` + `OPENAI_API_KEY`.

O contexto recuperado e as citações continuam disponíveis para inspeção."""
