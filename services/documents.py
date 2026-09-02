from db import get_connection

def list_documents(org_id):
    with get_connection() as c:
        rows = c.execute(
            "SELECT * FROM documents WHERE organization_id=? ORDER BY id DESC", (org_id,)
        ).fetchall()
    return [dict(x) for x in rows]
