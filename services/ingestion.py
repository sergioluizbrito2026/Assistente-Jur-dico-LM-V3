```python
"""
Assistente Jurídico SaaS IA V3
services/ingestion.py

Pipeline de ingestão documental.

Fluxo:

    Upload
       ↓
    Validação
       ↓
    Extração
       ↓
    OCR
       ↓
    Normalização
       ↓
    Chunking
       ↓
    SQLite
       ↓
    Embeddings
       ↓
    FAISS
       ↓
    Indexação
       ↓
    Status

Características:
- PDF
- DOCX
- TXT
- OCR opcional
- Chunking com overlap
- Persistência transacional
- Controle de duplicidade
- Isolamento por organização
- Indexação FAISS
- Status de processamento
- Tratamento de erros
- Compatível com services.embeddings
- Compatível com documents.py V3
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence, Tuple
import io
import json
import re

from db import get_connection
from services.embeddings import upsert_document_index


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

MAX_CHUNK_SIZE = 5000

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".docx",
    ".pdf",
}


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalize(text: str) -> str:
    """
    Normaliza espaços sem alterar significativamente
    o conteúdo jurídico.
    """

    text = text or ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# VALIDAÇÃO
# ============================================================

def _validate_org_id(
    org_id: Any,
) -> int:
    """
    Valida o ID da organização.
    """

    try:
        org_id = int(org_id)
    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "Organização inválida."
        )

    if org_id <= 0:
        raise ValueError(
            "Organização inválida."
        )

    return org_id


def _validate_filename(
    filename: str,
) -> str:
    """
    Valida o nome do arquivo.
    """

    filename = (
        filename or ""
    ).strip()

    if not filename:
        raise ValueError(
            "Nome do arquivo não informado."
        )

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        supported = ", ".join(
            sorted(
                SUPPORTED_EXTENSIONS
            )
        )

        raise ValueError(
            "Formato não suportado. "
            f"Utilize: {supported}."
        )

    return filename


# ============================================================
# CHUNKING
# ============================================================

def chunk_pages(
    page_texts: Sequence[Tuple[int, str]],
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[dict]:
    """
    Divide o conteúdo em chunks mantendo a página de origem.

    O overlap preserva contexto entre blocos.
    """

    try:
        size = int(size)
    except (
        TypeError,
        ValueError,
    ):
        size = DEFAULT_CHUNK_SIZE

    try:
        overlap = int(overlap)
    except (
        TypeError,
        ValueError,
    ):
        overlap = DEFAULT_CHUNK_OVERLAP

    if size <= 0:
        raise ValueError(
            "O tamanho do chunk deve ser maior que zero."
        )

    if size > MAX_CHUNK_SIZE:
        size = MAX_CHUNK_SIZE

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "O overlap deve ser >= 0 "
            "e menor que o tamanho do chunk."
        )

    chunks: List[dict] = []

    for page, text in page_texts:

        text = normalize(text)

        if not text:
            continue

        start = 0

        while start < len(text):

            end = min(
                len(text),
                start + size,
            )

            piece = text[
                start:end
            ].strip()

            if piece:

                chunks.append(
                    {
                        "page": page,
                        "content": piece,
                        "chunk_index": len(chunks),
                    }
                )

            if end >= len(text):
                break

            start = max(
                0,
                end - overlap,
            )

    return chunks


# ============================================================
# EXTRAÇÃO TXT
# ============================================================

def _extract_txt(
    data: bytes,
) -> List[Tuple[int, str]]:
    """
    Extrai texto de arquivo TXT.
    """

    text = data.decode(
        "utf-8",
        "ignore",
    )

    return [
        (1, text)
    ]


# ============================================================
# EXTRAÇÃO DOCX
# ============================================================

def _extract_docx(
    data: bytes,
) -> List[Tuple[int, str]]:
    """
    Extrai texto de DOCX.
    """

    from docx import Document

    doc = Document(
        io.BytesIO(data)
    )

    paragraphs = []

    for paragraph in doc.paragraphs:

        text = (
            paragraph.text or ""
        ).strip()

        if text:
            paragraphs.append(
                text
            )

    text = "\n".join(
        paragraphs
    )

    return [
        (1, text)
    ]


# ============================================================
# OCR
# ============================================================

def _ocr_page(
    page,
) -> str:
    """
    Executa OCR em uma página PDF.
    """

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(
        matrix=__import__(
            "fitz"
        ).Matrix(
            2,
            2,
        ),
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    )

    try:

        return pytesseract.image_to_string(
            image,
            lang="por+eng",
        )

    finally:

        image.close()


# ============================================================
# EXTRAÇÃO PDF
# ============================================================

def _extract_pdf(
    data: bytes,
    use_ocr: bool = True,
) -> Tuple[List[Tuple[int, str]], int]:
    """
    Extrai texto de PDF.

    Utiliza extração nativa primeiro.
    OCR somente quando necessário.
    """

    import fitz

    pages: List[
        Tuple[int, str]
    ] = []

    ocr_pages = 0

    pdf = fitz.open(
        stream=data,
        filetype="pdf",
    )

    try:

        for page_number, page in enumerate(
            pdf,
            start=1,
        ):

            text = (
                page.get_text("text")
                or ""
            ).strip()

            # OCR somente quando
            # praticamente não existe texto.
            if (
                len(text) < 30
                and use_ocr
            ):

                try:

                    text = _ocr_page(
                        page
                    )

                    ocr_pages += 1

                except Exception:
                    # Mantém a extração original.
                    pass

            pages.append(
                (
                    page_number,
                    text,
                )
            )

    finally:

        pdf.close()

    return pages, ocr_pages


# ============================================================
# EXTRAÇÃO PRINCIPAL
# ============================================================

def extract_pages(
    uploaded,
    use_ocr: bool = True,
):
    """
    Extrai texto de TXT, DOCX e PDF.

    Retorna:

        pages
        ocr_pages
    """

    if uploaded is None:
        raise ValueError(
            "Arquivo não informado."
        )

    data = uploaded.read()

    if not data:
        raise ValueError(
            "O arquivo enviado está vazio."
        )

    filename = str(
        getattr(
            uploaded,
            "name",
            "",
        )
    ).lower()

    extension = Path(
        filename
    ).suffix.lower()

    if extension == ".txt":

        return (
            _extract_txt(data),
            0,
        )

    if extension == ".docx":

        return (
            _extract_docx(data),
            0,
        )

    if extension == ".pdf":

        return _extract_pdf(
            data,
            use_ocr=use_ocr,
        )

    raise ValueError(
        "Formato não suportado. "
        "Utilize arquivos TXT, DOCX ou PDF."
    )


# ============================================================
# VERIFICA DUPLICIDADE
# ============================================================

def _document_exists(
    org_id: int,
    filename: str,
) -> bool:
    """
    Verifica se o documento já existe na organização.
    """

    with get_connection() as c:

        row = c.execute(
            """
            SELECT id
            FROM documents
            WHERE
                organization_id = ?
                AND name = ?
            LIMIT 1
            """,
            (
                org_id,
                filename,
            ),
        ).fetchone()

    return row is not None


# ============================================================
# SALVAR DOCUMENTO E CHUNKS
# ============================================================

def _save_document(
    org_id: int,
    filename: str,
    extension: str,
    pages: Sequence[Tuple[int, str]],
    chunks: Sequence[dict],
    ocr_pages: int,
) -> int:
    """
    Persiste documento e chunks em uma transação.
    """

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    with get_connection() as c:

        existing = c.execute(
            """
            SELECT id
            FROM documents
            WHERE
                organization_id = ?
                AND name = ?
            LIMIT 1
            """,
            (
                org_id,
                filename,
            ),
        ).fetchone()

        if existing:

            raise ValueError(
                f"O documento '{filename}' "
                "já está cadastrado."
            )

        cursor = c.execute(
            """
            INSERT INTO documents(
                organization_id,
                name,
                type,
                status,
                pages,
                chunks,
                ocr_pages,
                created_at
            )
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                org_id,
                filename,
                extension,
                "Processando",
                len(pages),
                len(chunks),
                ocr_pages,
                now,
            ),
        )

        document_id = int(
            cursor.lastrowid
        )

        for chunk in chunks:

            content = (
                chunk.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            metadata = {
                "source": filename,
                "page": chunk.get(
                    "page"
                ),
                "chunk_index": chunk.get(
                    "chunk_index"
                ),
                "ingested_at": now,
            }

            c.execute(
                """
                INSERT INTO chunks(
                    document_id,
                    organization_id,
                    content,
                    page,
                    chunk_index,
                    token_estimate,
                    metadata
                )
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    document_id,
                    org_id,
                    content,
                    chunk.get("page"),
                    chunk.get(
                        "chunk_index"
                    ),
                    max(
                        1,
                        len(content) // 4,
                    ),
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                ),
            )

    return document_id


# ============================================================
# ATUALIZAR STATUS
# ============================================================

def _update_document_status(
    document_id: int,
    status: str,
) -> None:
    """
    Atualiza o status do documento.
    """

    with get_connection() as c:

        c.execute(
            """
            UPDATE documents
            SET status = ?
            WHERE id = ?
            """,
            (
                status,
                document_id,
            ),
        )


# ============================================================
# INGESTÃO PRINCIPAL
# ============================================================

def ingest_document(
    uploaded,
    org_id,
    use_ocr: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    """
    Executa a ingestão completa.

    Pipeline:

        Upload
           ↓
        Extração
           ↓
        OCR
           ↓
        Chunking
           ↓
        SQLite
           ↓
        Embeddings
           ↓
        FAISS
           ↓
        Indexado
    """

    org_id = _validate_org_id(
        org_id
    )

    filename = _validate_filename(
        str(
            getattr(
                uploaded,
                "name",
                "",
            )
        )
    )

    # --------------------------------------------------------
    # Duplicidade
    # --------------------------------------------------------

    if _document_exists(
        org_id,
        filename,
    ):

        raise ValueError(
            f"O documento '{filename}' "
            "já está cadastrado."
        )

    # --------------------------------------------------------
    # Extração
    # --------------------------------------------------------

    pages, ocr_pages = extract_pages(
        uploaded,
        use_ocr=use_ocr,
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunks = chunk_pages(
        pages,
        size=chunk_size,
        overlap=chunk_overlap,
    )

    if not chunks:

        raise ValueError(
            "Não foi possível extrair "
            "texto do documento."
        )

    # --------------------------------------------------------
    # Persistência
    # --------------------------------------------------------

    extension = (
        Path(filename)
        .suffix
        .upper()
        .replace(
            ".",
            "",
        )
    )

    document_id = _save_document(
        org_id=org_id,
        filename=filename,
        extension=extension,
        pages=pages,
        chunks=chunks,
        ocr_pages=ocr_pages,
    )

    # --------------------------------------------------------
    # Indexação FAISS
    # --------------------------------------------------------

    try:

        indexed_chunks = upsert_document_index(
            org_id,
            document_id,
        )

    except Exception as exc:

        _update_document_status(
            document_id,
            "Erro na indexação",
        )

        raise RuntimeError(
            "Documento salvo, mas a "
            "indexação vetorial falhou: "
            f"{exc}"
        ) from exc

    # --------------------------------------------------------
    # Status final
    # --------------------------------------------------------

    _update_document_status(
        document_id,
        "Indexado",
    )

    return {
        "document_id": document_id,
        "chunks": len(chunks),
        "indexed_chunks": indexed_chunks,
        "pages": len(pages),
        "ocr_pages": ocr_pages,
        "status": "Indexado",
    }


# ============================================================
# REINDEXAÇÃO
# ============================================================

def reindex_document(
    document_id: int,
    org_id: int,
) -> dict:
    """
    Reindexa um documento já existente.

    Útil quando:
    - o FAISS foi apagado;
    - o modelo de embeddings mudou;
    - a indexação anterior falhou.
    """

    try:
        document_id = int(
            document_id
        )
        org_id = int(
            org_id
        )
    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "IDs inválidos."
        )

    with get_connection() as c:

        row = c.execute(
            """
            SELECT id, name
            FROM documents
            WHERE
                id = ?
                AND organization_id = ?
            LIMIT 1
            """,
            (
                document_id,
                org_id,
            ),
        ).fetchone()

    if row is None:

        raise ValueError(
            "Documento não encontrado."
        )

    _update_document_status(
        document_id,
        "Processando",
    )

    try:

        indexed = upsert_document_index(
            org_id,
            document_id,
        )

        _update_document_status(
            document_id,
            "Indexado",
        )

        return {
            "document_id": document_id,
            "indexed_chunks": indexed,
            "status": "Indexado",
        }

    except Exception as exc:

        _update_document_status(
            document_id,
            "Erro na indexação",
        )

        raise RuntimeError(
            f"Falha ao reindexar documento: {exc}"
        ) from exc


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> dict:
    """
    Teste estrutural.

    Não executa OCR, embeddings ou FAISS.
    """

    required = [
        "normalize",
        "chunk_pages",
        "extract_pages",
        "ingest_document",
        "reindex_document",
    ]

    missing = [
        name
        for name in required
        if name not in globals()
    ]

    return {
        "module": "services.ingestion",
        "status": (
            "ok"
            if not missing
            else "error"
        ),
        "required_functions": required,
        "missing_functions": missing,
        "supported_extensions": sorted(
            SUPPORTED_EXTENSIONS
        ),
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    result = self_test()

    print("=" * 60)
    print("INGESTION.PY V3 - SELF TEST")
    print("=" * 60)

    print(
        f"Status: {result['status']}"
    )

    print(
        f"Formatos: "
        f"{result['supported_extensions']}"
    )

    print(
        f"Chunk size: "
        f"{result['default_chunk_size']}"
    )

    print(
        f"Overlap: "
        f"{result['default_chunk_overlap']}"
    )

    print(
        f"Funções ausentes: "
        f"{result['missing_functions']}"
    )

    print("=" * 60)
```
