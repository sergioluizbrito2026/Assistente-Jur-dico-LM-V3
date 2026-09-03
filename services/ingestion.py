from datetime import datetime
from pathlib import Path
import io
import json
import re

from db import get_connection
from services.embeddings import upsert_document_index


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalize(text):
    """
    Normaliza espaços sem alterar o conteúdo jurídico.
    """
    return re.sub(r"\s+", " ", text or "").strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_pages(page_texts, size=1000, overlap=150):
    """
    Divide o conteúdo em chunks mantendo a página de origem.

    O overlap ajuda o RAG a preservar contexto entre blocos.
    """
    if size <= 0:
        raise ValueError("O tamanho do chunk deve ser maior que zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError("O overlap deve ser >= 0 e menor que o tamanho do chunk.")

    chunks = []

    for page, text in page_texts:
        text = normalize(text)

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(len(text), start + size)
            piece = text[start:end].strip()

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

            start = max(0, end - overlap)

    return chunks


# ============================================================
# EXTRAÇÃO DE DOCUMENTOS
# ============================================================

def extract_pages(uploaded, use_ocr=True):
    """
    Extrai texto de TXT, DOCX e PDF.

    PDF:
    - tenta extração nativa primeiro;
    - utiliza OCR apenas quando a página possui pouco texto.
    """

    data = uploaded.read()

    if not data:
        raise ValueError("O arquivo enviado está vazio.")

    name = str(getattr(uploaded, "name", "")).lower()

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if name.endswith(".txt"):
        text = data.decode("utf-8", "ignore")
        return [(1, text)], 0

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in doc.paragraphs
            if paragraph.text and paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        return [(1, text)], 0

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if name.endswith(".pdf"):
        import fitz

        pages = []
        ocr_pages = 0

        pdf = fitz.open(stream=data, filetype="pdf")

        try:
            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                # OCR somente quando necessário
                if len(text) < 30 and use_ocr:
                    import pytesseract
                    from PIL import Image

                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )

                    image = Image.open(
                        io.BytesIO(pix.tobytes("png"))
                    )

                    try:
                        text = pytesseract.image_to_string(
                            image,
                            lang="por+eng",
                        )
                    finally:
                        image.close()

                    ocr_pages += 1

                pages.append((page_number, text))

        finally:
            pdf.close()

        return pages, ocr_pages

    raise ValueError(
        "Formato não suportado. Utilize arquivos TXT, DOCX ou PDF."
    )


# ============================================================
# INGESTÃO
# ============================================================

def ingest_document(uploaded, org_id, use_ocr=True):
    """
    Pipeline:

    Upload
       ↓
    Extração
       ↓
    OCR quando necessário
       ↓
    Chunking
       ↓
    SQLite
       ↓
    Embeddings / FAISS
       ↓
    Status Indexado
    """

    if not org_id:
        raise ValueError("Organização não informada.")

    filename = str(getattr(uploaded, "name", "")).strip()

    if not filename:
        raise ValueError("Nome do arquivo não informado.")

    # --------------------------------------------------------
    # Extração
    # --------------------------------------------------------

    pages, ocr_pages = extract_pages(
        uploaded,
        use_ocr=use_ocr,
    )

    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError(
            "Não foi possível extrair texto do documento."
        )

    now = datetime.now().isoformat(timespec="seconds")

    extension = Path(filename).suffix.upper().replace(".", "")

    document_id = None

    try:
        # ----------------------------------------------------
        # Persistência transacional
        # ----------------------------------------------------

        with get_connection() as c:

            # Evita duplicação acidental do mesmo documento
            existing = c.execute(
                """
                SELECT id
                FROM documents
                WHERE organization_id = ?
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
                    f"O documento '{filename}' já está cadastrado."
                )

            cur = c.execute(
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

            document_id = cur.lastrowid

            for chunk in chunks:

                metadata = {
                    "source": filename,
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
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
                        chunk["content"],
                        chunk["page"],
                        chunk["chunk_index"],
                        max(
                            1,
                            len(chunk["content"]) // 4,
                        ),
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                        ),
                    ),
                )

        # ----------------------------------------------------
        # Indexação vetorial
        # ----------------------------------------------------

        try:
            upsert_document_index(
                org_id,
                document_id,
            )

        except Exception as exc:

            # Mantém o documento no banco, mas deixa explícito
            # que a indexação vetorial falhou.
            with get_connection() as c:
                c.execute(
                    """
                    UPDATE documents
                    SET status = ?
                    WHERE id = ?
                    """,
                    (
                        "Erro na indexação",
                        document_id,
                    ),
                )

            raise RuntimeError(
                f"Documento salvo, mas a indexação vetorial falhou: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Indexação concluída
        # ----------------------------------------------------

        with get_connection() as c:
            c.execute(
                """
                UPDATE documents
                SET status = ?
                WHERE id = ?
                """,
                (
                    "Indexado",
                    document_id,
                ),
            )

        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "pages": len(pages),
            "ocr_pages": ocr_pages,
            "status": "Indexado",
        }

    except Exception:
        # Se a falha ocorrer antes da criação/indexação,
        # a transação do SQLite é revertida automaticamente.
        raise
