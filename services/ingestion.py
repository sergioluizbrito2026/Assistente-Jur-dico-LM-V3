from datetime import datetime
from pathlib import Path
import io, re, json
from db import get_connection
from services.embeddings import upsert_document_index

def normalize(text):
    return re.sub(r"\s+"," ",text or "").strip()

def chunk_pages(page_texts, size=1000, overlap=150):
    chunks=[]
    for page, text in page_texts:
        text=normalize(text)
        if not text:
            continue
        start=0
        while start < len(text):
            end=min(len(text),start+size)
            piece=text[start:end]
            chunks.append({"page":page,"content":piece,"chunk_index":len(chunks)})
            if end>=len(text):
                break
            start=max(0,end-overlap)
    return chunks

def extract_pages(uploaded, use_ocr=True):
    data=uploaded.read()
    name=uploaded.name.lower()
    if name.endswith(".txt"):
        return [(1,data.decode("utf-8","ignore"))],0
    if name.endswith(".docx"):
        from docx import Document
        doc=Document(io.BytesIO(data))
        text="\n".join(p.text for p in doc.paragraphs)
        return [(1,text)],0
    if name.endswith(".pdf"):
        import fitz
        import pytesseract
        from PIL import Image
        pdf=fitz.open(stream=data,filetype="pdf")
        pages=[]; ocr_pages=0
        for i,page in enumerate(pdf,1):
            text=page.get_text("text").strip()
            if len(text)<30 and use_ocr:
                pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
                img=Image.open(io.BytesIO(pix.tobytes("png")))
                text=pytesseract.image_to_string(img,lang="por+eng")
                ocr_pages+=1
            pages.append((i,text))
        return pages,ocr_pages
    raise ValueError("Formato não suportado.")

def ingest_document(uploaded, org_id, use_ocr=True):
    pages,ocr_pages=extract_pages(uploaded,use_ocr)
    chunks=chunk_pages(pages)
    if not chunks:
        raise ValueError("Não foi possível extrair texto do documento.")

    now=datetime.now().isoformat(timespec="seconds")
    with get_connection() as c:
        cur=c.execute(
            "INSERT INTO documents(organization_id,name,type,status,pages,chunks,ocr_pages,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (org_id,uploaded.name,Path(uploaded.name).suffix.upper().replace(".",""),
             "Processando",len(pages),len(chunks),ocr_pages,now)
        )
        did=cur.lastrowid
        for ch in chunks:
            c.execute(
                "INSERT INTO chunks(document_id,organization_id,content,page,chunk_index,token_estimate,metadata) VALUES(?,?,?,?,?,?,?)",
                (did,org_id,ch["content"],ch["page"],ch["chunk_index"],
                 max(1,len(ch["content"])//4),json.dumps({"source":uploaded.name},ensure_ascii=False))
            )
        c.execute("UPDATE documents SET status='Indexado' WHERE id=?",(did,))

    upsert_document_index(org_id,did)
    return {"document_id":did,"chunks":len(chunks),"pages":len(pages),"ocr_pages":ocr_pages}
