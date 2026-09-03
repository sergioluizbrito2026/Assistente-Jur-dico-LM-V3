"""
Assistente Jurídico SaaS IA V3
services/evaluation.py

Módulo de avaliação determinística das respostas do RAG.

Métricas principais:

1. Context Relevance
   Mede o quanto os chunks recuperados são relevantes para a pergunta.

2. Citation Coverage
   Mede a presença e validade das citações utilizadas pela resposta.

3. Groundedness
   Mede o quanto as frases da resposta apresentam evidências
   lexicais compatíveis com o contexto recuperado.

4. Overall
   Score geral ponderado das métricas anteriores.

Características:
- Não depende de LLM.
- Não depende de FAISS.
- Não depende de Streamlit.
- Não altera a resposta da IA.
- Aceita diferentes formatos de chunks/citações.
- Trata entradas vazias ou inválidas.
- Compatível com o RAG Pipeline V3.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MIN_TOKEN_LEN = 3

WEIGHT_CONTEXT = 0.40
WEIGHT_CITATIONS = 0.25
WEIGHT_GROUNDEDNESS = 0.35


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    # Português
    "a",
    "à",
    "às",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "é",
    "em",
    "entre",
    "essa",
    "esse",
    "esta",
    "este",
    "estas",
    "estes",
    "foi",
    "for",
    "há",
    "isso",
    "isto",
    "já",
    "mais",
    "mas",
    "mesmo",
    "na",
    "nas",
    "não",
    "no",
    "nos",
    "num",
    "numa",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "quando",
    "que",
    "se",
    "sem",
    "ser",
    "são",
    "sua",
    "suas",
    "seu",
    "seus",
    "tem",
    "têm",
    "um",
    "uma",
    "umas",
    "uns",

    # Inglês
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "may",
    "into",
    "about",
    "what",
    "which",
    "where",
    "when",
}


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def _normalize_text(text: Any) -> str:
    """
    Normaliza texto para comparação.

    Remove diferenças de:
    - maiúsculas/minúsculas;
    - acentuação;
    - espaços laterais.

    Não modifica o texto original armazenado no sistema.
    """

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    return text.lower().strip()


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def _tokens(text: Any) -> set[str]:
    """
    Extrai tokens relevantes do texto.
    """

    normalized = _normalize_text(text)

    raw_tokens = re.findall(
        r"[a-z0-9]+",
        normalized,
    )

    return {
        token
        for token in raw_tokens
        if len(token) >= MIN_TOKEN_LEN
        and token not in STOPWORDS
    }


# ============================================================
# UTILITÁRIOS NUMÉRICOS
# ============================================================

def _clip(value: float) -> float:
    """
    Mantém o valor entre 0 e 1.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if value != value:
        return 0.0

    return max(
        0.0,
        min(1.0, value),
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Converte um valor para float com segurança.
    """

    try:
        return _clip(float(value))
    except (TypeError, ValueError):
        return default


# ============================================================
# EXTRAÇÃO DE CONTEÚDO DOS CHUNKS
# ============================================================

def _chunk_content(chunk: Any) -> str:
    """
    Extrai o conteúdo de um chunk.

    Compatível com estruturas como:

        {
            "content": "...",
        }

    ou:

        {
            "text": "...",
        }

    ou:

        {
            "page_content": "...",
        }
    """

    if chunk is None:
        return ""

    if isinstance(chunk, str):
        return chunk

    if not isinstance(chunk, dict):
        return str(chunk)

    possible_fields = (
        "content",
        "text",
        "chunk",
        "page_content",
        "document_content",
        "body",
    )

    for field in possible_fields:

        value = chunk.get(field)

        if value:
            return str(value)

    return ""


# ============================================================
# ID DE CHUNK
# ============================================================

def _chunk_id(chunk: Any) -> str:
    """
    Obtém o identificador do chunk.
    """

    if not isinstance(chunk, dict):
        return ""

    for field in (
        "chunk_id",
        "id",
        "chunk",
    ):
        value = chunk.get(field)

        if value is not None:
            return str(value)

    return ""


# ============================================================
# ID DO DOCUMENTO
# ============================================================

def _document_id(chunk: Any) -> str:
    """
    Obtém o identificador do documento.
    """

    if not isinstance(chunk, dict):
        return ""

    for field in (
        "document_id",
        "document",
        "document_name",
        "name",
    ):
        value = chunk.get(field)

        if value is not None:
            return str(value)

    return ""


# ============================================================
# SCORE DE SIMILARIDADE LEXICAL
# ============================================================

def _token_overlap(
    left: Any,
    right: Any,
) -> float:
    """
    Calcula sobreposição de tokens.

    Fórmula:

        tokens_em_comum / tokens_da_pergunta

    O objetivo é verificar se o contexto contém termos relevantes
    para a consulta.
    """

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = left_tokens.intersection(
        right_tokens
    )

    return _clip(
        len(intersection)
        / max(len(left_tokens), 1)
    )


# ============================================================
# CONTEXT RELEVANCE
# ============================================================

def context_relevance(
    question: str,
    chunks: Sequence[Any],
) -> float:
    """
    Mede a relevância do contexto recuperado.

    O cálculo combina:

    1. Sobreposição lexical da pergunta com os chunks.
    2. Score do retriever, quando disponível.
    3. Score do reranker, quando disponível.

    O reranker recebe maior prioridade quando existe.
    """

    question = (question or "").strip()

    if not question:
        return 0.0

    if not chunks:
        return 0.0

    question_tokens = _tokens(question)

    if not question_tokens:
        return 0.0

    scores: List[float] = []

    for chunk in chunks:

        content = _chunk_content(chunk)

        if not content:
            continue

        lexical = _token_overlap(
            question,
            content,
        )

        semantic_hint = 0.0

        if isinstance(chunk, dict):

            reranker_score = chunk.get(
                "reranker_score"
            )

            retriever_score = chunk.get(
                "retriever_score"
            )

            if reranker_score is not None:
                semantic_hint = _safe_float(
                    reranker_score
                )

            elif retriever_score is not None:
                semantic_hint = _safe_float(
                    retriever_score
                )

        # Se houver score externo, combina com a evidência lexical.
        if semantic_hint > 0:
            score = (
                lexical * 0.70
                + semantic_hint * 0.30
            )
        else:
            score = lexical

        scores.append(
            _clip(score)
        )

    if not scores:
        return 0.0

    # Prioriza os melhores contextos.
    scores.sort(reverse=True)

    top_scores = scores[:5]

    return _clip(
        sum(top_scores)
        / len(top_scores)
    )


# ============================================================
# EXTRAÇÃO DE CITAÇÕES DA RESPOSTA
# ============================================================

def _extract_citation_ids(
    answer: str,
) -> List[str]:
    """
    Extrai citações no padrão:

        [1]
        [2]
        [10]

    Também reconhece:

        [1, 2]
        [1][2]
    """

    if not answer:
        return []

    matches = re.findall(
        r"\[(\d+)\]",
        str(answer),
    )

    result: List[str] = []

    for value in matches:

        if value not in result:
            result.append(value)

    return result


# ============================================================
# NORMALIZAÇÃO DE CITAÇÕES
# ============================================================

def _citation_id(
    citation: Any,
    fallback: int,
) -> str:
    """
    Obtém o ID de uma citação.
    """

    if isinstance(citation, dict):

        value = citation.get("id")

        if value is not None:
            return str(value)

        value = citation.get(
            "citation_id"
        )

        if value is not None:
            return str(value)

    elif citation is not None:

        return str(citation)

    return str(fallback)


def _citation_content(
    citation: Any,
) -> str:
    """
    Obtém conteúdo associado à citação.
    """

    if isinstance(citation, str):
        return citation

    if not isinstance(citation, dict):
        return ""

    fields = (
        "content",
        "text",
        "excerpt",
        "quote",
        "evidence",
        "page_content",
    )

    for field in fields:

        value = citation.get(field)

        if value:
            return str(value)

    return ""


# ============================================================
# CITATION COVERAGE
# ============================================================

def citation_coverage(
    answer: str,
    citations: Sequence[Any],
) -> float:
    """
    Mede a cobertura das citações.

    A métrica considera:

    - quantidade de referências presentes na resposta;
    - quantidade de referências válidas;
    - cobertura do conjunto de evidências disponíveis.

    Se não houver citações disponíveis, retorna 0.
    """

    answer = answer or ""

    if not citations:
        return 0.0

    citation_ids = {
        _citation_id(
            citation,
            index + 1,
        )
        for index, citation in enumerate(citations)
    }

    if not citation_ids:
        return 0.0

    used_ids = _extract_citation_ids(
        answer
    )

    if not used_ids:
        return 0.0

    valid_ids = [
        citation_id
        for citation_id in used_ids
        if citation_id in citation_ids
    ]

    if not valid_ids:
        return 0.0

    # Precisão das referências usadas.
    precision = (
        len(valid_ids)
        / max(len(used_ids), 1)
    )

    # Cobertura das evidências disponíveis.
    coverage = (
        len(set(valid_ids))
        / max(len(citation_ids), 1)
    )

    # Evita que apenas uma citação de um conjunto enorme
    # seja tratada como cobertura perfeita.
    score = (
        precision * 0.60
        + coverage * 0.40
    )

    return _clip(score)


# ============================================================
# DIVISÃO EM SENTENÇAS
# ============================================================

def _split_sentences(
    text: str,
) -> List[str]:
    """
    Divide uma resposta em sentenças.

    Mantém frases com tamanho suficiente para avaliação.
    """

    if not text:
        return []

    text = str(text).strip()

    if not text:
        return []

    # Remove blocos de markdown excessivamente curtos.
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    sentences = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        # Ignora somente marcadores.
        if re.fullmatch(
            r"[-*•#\s]+",
            part,
        ):
            continue

        if len(_tokens(part)) >= 2:
            sentences.append(part)

    return sentences


# ============================================================
# GROUNDEDNESS
# ============================================================

def groundedness(
    answer: str,
    chunks: Sequence[Any],
) -> float:
    """
    Estima groundedness por suporte lexical.

    Para cada sentença da resposta:

        tokens da sentença
        versus
        tokens do contexto recuperado.

    Uma sentença é considerada suportada quando existe
    sobreposição suficiente com o contexto.

    Isto é uma heurística determinística, não uma prova
    semântica absoluta.
    """

    if not answer:
        return 0.0

    if not chunks:
        return 0.0

    context_parts = [
        _chunk_content(chunk)
        for chunk in chunks
    ]

    context_parts = [
        content
        for content in context_parts
        if content
    ]

    if not context_parts:
        return 0.0

    full_context = " ".join(
        context_parts
    )

    sentences = _split_sentences(
        answer
    )

    if not sentences:
        return 0.0

    sentence_scores: List[float] = []

    context_tokens = _tokens(
        full_context
    )

    if not context_tokens:
        return 0.0

    for sentence in sentences:

        sentence_tokens = _tokens(
            sentence
        )

        if not sentence_tokens:
            continue

        overlap = (
            len(
                sentence_tokens.intersection(
                    context_tokens
                )
            )
            / max(
                len(sentence_tokens),
                1,
            )
        )

        # Procura também o melhor chunk individual.
        best_chunk_overlap = 0.0

        for content in context_parts:

            chunk_tokens = _tokens(
                content
            )

            if not chunk_tokens:
                continue

            local_overlap = (
                len(
                    sentence_tokens.intersection(
                        chunk_tokens
                    )
                )
                / max(
                    len(sentence_tokens),
                    1,
                )
            )

            best_chunk_overlap = max(
                best_chunk_overlap,
                local_overlap,
            )

        score = max(
            overlap,
            best_chunk_overlap,
        )

        sentence_scores.append(
            _clip(score)
        )

    if not sentence_scores:
        return 0.0

    return _clip(
        sum(sentence_scores)
        / len(sentence_scores)
    )


# ============================================================
# SUPORTE POR SENTENÇA
# ============================================================

def _supported_sentences(
    answer: str,
    chunks: Sequence[Any],
) -> Tuple[int, int]:
    """
    Retorna:

        (sentenças suportadas, total de sentenças)
    """

    sentences = _split_sentences(
        answer
    )

    if not sentences or not chunks:
        return 0, len(sentences)

    context = " ".join(
        _chunk_content(chunk)
        for chunk in chunks
    )

    context_tokens = _tokens(
        context
    )

    if not context_tokens:
        return 0, len(sentences)

    supported = 0

    for sentence in sentences:

        sentence_tokens = _tokens(
            sentence
        )

        if not sentence_tokens:
            continue

        overlap = (
            len(
                sentence_tokens.intersection(
                    context_tokens
                )
            )
            / max(
                len(sentence_tokens),
                1,
            )
        )

        # 15% de tokens relevantes em comum
        # funciona como limiar conservador.
        if overlap >= 0.15:
            supported += 1

    return supported, len(sentences)


# ============================================================
# SCORE DE QUALIDADE
# ============================================================

def _quality_label(
    score: float,
) -> str:
    """
    Classifica o score geral.
    """

    score = _clip(score)

    if score >= 0.85:
        return "Excelente"

    if score >= 0.70:
        return "Bom"

    if score >= 0.50:
        return "Atenção"

    return "Baixo"


# ============================================================
# RECOMENDAÇÕES
# ============================================================

def _build_recommendations(
    context_score: float,
    citation_score: float,
    grounded_score: float,
) -> List[str]:
    """
    Gera recomendações simples para diagnóstico.
    """

    recommendations: List[str] = []

    if context_score < 0.50:
        recommendations.append(
            "O contexto recuperado apresenta baixa relevância. "
            "Revise a consulta, o chunking, os embeddings e o reranker."
        )

    if citation_score < 0.50:
        recommendations.append(
            "A resposta apresenta baixa cobertura de citações. "
            "Verifique se as evidências recuperadas estão sendo referenciadas."
        )

    if grounded_score < 0.50:
        recommendations.append(
            "A resposta possui baixo suporte lexical no contexto. "
            "Revise o prompt de groundedness e evite informações não presentes "
            "nas evidências recuperadas."
        )

    if not recommendations:
        recommendations.append(
            "Resposta com bons sinais de relevância, citação e fundamentação."
        )

    return recommendations


# ============================================================
# AVALIAÇÃO COMPLETA
# ============================================================

def evaluate_answer(
    question: str,
    answer: str,
    chunks: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Executa a avaliação completa.

    Compatível com:

        evaluate_answer(
            question,
            answer,
            result["reranked"],
            result["citations"],
        )

    Retorna:

        {
            "context_relevance": 0.00,
            "citation_coverage": 0.00,
            "groundedness": 0.00,
            "overall": 0.00,
            ...
        }
    """

    question = str(
        question or ""
    ).strip()

    answer = str(
        answer or ""
    ).strip()

    chunks = list(
        chunks or []
    )

    citations = list(
        citations or []
    )

    # --------------------------------------------------------
    # Entradas inválidas
    # --------------------------------------------------------

    if not question:

        return {
            "context_relevance": 0.0,
            "citation_coverage": 0.0,
            "groundedness": 0.0,
            "overall": 0.0,
            "quality": "Baixo",
            "valid": False,
            "reason": "Pergunta vazia.",
            "context_count": len(chunks),
            "citation_count": len(citations),
            "citation_ids_used": [],
            "supported_sentences": 0,
            "total_sentences": 0,
            "recommendations": [
                "Informe uma pergunta para executar a avaliação."
            ],
        }

    if not answer:

        return {
            "context_relevance": context_relevance(
                question,
                chunks,
            ),
            "citation_coverage": 0.0,
            "groundedness": 0.0,
            "overall": 0.0,
            "quality": "Baixo",
            "valid": False,
            "reason": "Resposta vazia.",
            "context_count": len(chunks),
            "citation_count": len(citations),
            "citation_ids_used": [],
            "supported_sentences": 0,
            "total_sentences": 0,
            "recommendations": [
                "Informe uma resposta para executar a avaliação."
            ],
        }

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    context_score = context_relevance(
        question,
        chunks,
    )

    citation_score = citation_coverage(
        answer,
        citations,
    )

    grounded_score = groundedness(
        answer,
        chunks,
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = (
        context_score * WEIGHT_CONTEXT
        + citation_score * WEIGHT_CITATIONS
        + grounded_score * WEIGHT_GROUNDEDNESS
    )

    overall = _clip(
        overall
    )

    # --------------------------------------------------------
    # Sentenças suportadas
    # --------------------------------------------------------

    supported, total = _supported_sentences(
        answer,
        chunks,
    )

    # --------------------------------------------------------
    # Citações utilizadas
    # --------------------------------------------------------

    used_citation_ids = _extract_citation_ids(
        answer
    )

    # --------------------------------------------------------
    # Recomendações
    # --------------------------------------------------------

    recommendations = _build_recommendations(
        context_score,
        citation_score,
        grounded_score,
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    return {
        # Scores principais
        "context_relevance": round(
            context_score,
            4,
        ),
        "citation_coverage": round(
            citation_score,
            4,
        ),
        "groundedness": round(
            grounded_score,
            4,
        ),
        "overall": round(
            overall,
            4,
        ),

        # Classificação
        "quality": _quality_label(
            overall
        ),

        # Estado
        "valid": True,

        # Diagnóstico
        "context_count": len(
            chunks
        ),

        "citation_count": len(
            citations
        ),

        "citation_ids_used": used_citation_ids,

        "supported_sentences": supported,

        "total_sentences": total,

        "recommendations": recommendations,

        # Configuração utilizada
        "weights": {
            "context_relevance": WEIGHT_CONTEXT,
            "citation_coverage": WEIGHT_CITATIONS,
            "groundedness": WEIGHT_GROUNDEDNESS,
        },
    }


# ============================================================
# ALIASES
# ============================================================

def evaluate_rag(
    question: str,
    answer: str,
    chunks: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Alias para integração futura.
    """

    return evaluate_answer(
        question,
        answer,
        chunks,
        citations,
    )


def evaluate_response(
    question: str,
    answer: str,
    chunks: Sequence[Any] | None = None,
    citations: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """
    Alias para integração futura.
    """

    return evaluate_answer(
        question,
        answer,
        chunks,
        citations,
    )


# ============================================================
# TESTE INTERNO
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Executa um teste rápido do módulo sem depender de banco,
    FAISS, embeddings ou LLM.

    Útil para verificar se o evaluation.py está funcionando.
    """

    question = (
        "Qual é o prazo para apresentação da contestação?"
    )

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "document": "peticao.pdf",
            "page": 3,
            "content": (
                "O prazo para apresentação da contestação "
                "será de 15 dias úteis, contados conforme "
                "a legislação processual aplicável."
            ),
            "retriever_score": 0.91,
            "reranker_score": 0.94,
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-001",
            "document": "peticao.pdf",
            "page": 4,
            "content": (
                "A parte deverá observar as regras de "
                "intimação e contagem dos prazos processuais."
            ),
            "retriever_score": 0.75,
            "reranker_score": 0.82,
        },
    ]

    citations = [
        {
            "id": 1,
            "document": "peticao.pdf",
            "page": 3,
            "chunk_id": "chunk-001",
            "content": chunks[0]["content"],
        },
        {
            "id": 2,
            "document": "peticao.pdf",
            "page": 4,
            "chunk_id": "chunk-002",
            "content": chunks[1]["content"],
        },
    ]

    answer = (
        "Conforme o documento, o prazo para apresentação "
        "da contestação é de 15 dias úteis. [1] "
        "A contagem deve observar as regras de intimação "
        "e dos prazos processuais. [2]"
    )

    result = evaluate_answer(
        question,
        answer,
        chunks,
        citations,
    )

    return result


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print("EVALUATION.PY V3 - SELF TEST")
    print("=" * 60)

    print(
        f"Context relevance : "
        f"{result['context_relevance']:.4f}"
    )

    print(
        f"Citation coverage : "
        f"{result['citation_coverage']:.4f}"
    )

    print(
        f"Groundedness      : "
        f"{result['groundedness']:.4f}"
    )

    print(
        f"Overall            : "
        f"{result['overall']:.4f}"
    )

    print(
        f"Quality            : "
        f"{result['quality']}"
    )

    print(
        f"Contextos         : "
        f"{result['context_count']}"
    )

    print(
        f"Citações          : "
        f"{result['citation_count']}"
    )

    print(
        f"Sentenças         : "
        f"{result['supported_sentences']}/"
        f"{result['total_sentences']}"
    )

    print(
        f"Citações usadas   : "
        f"{result['citation_ids_used']}"
    )

    print("=" * 60)
