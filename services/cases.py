from datetime import datetime
from db import get_connection

def create_case(org,title,client,category,priority):
    with get_connection() as c:
        c.execute(
            "INSERT INTO cases(organization_id,title,client,category,priority,created_at) VALUES(?,?,?,?,?,?)",
            (org,title,client,category,priority,datetime.now().isoformat(timespec="seconds"))
        )

def list_cases(org):
    with get_connection() as c:
        return [dict(x) for x in c.execute(
            "SELECT * FROM cases WHERE organization_id=? ORDER BY id DESC",(org,)
        )]
