from datetime import datetime
import json
from db import get_connection


def audit(user, action, entity_type, entity_id, metadata=None):
  with get_connection() as c:
    c.execute(
        "INSERT INTO audit_logs(organization_id,user_id,action,entity_type,entity_id,metadata,created_at) VALUES(?,?,?,?,?,?,?)",
        (
            user["organization_id"],
            user["id"],
            action,
            entity_type,
            entity_id,
            json.dumps(metadata or {}, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
