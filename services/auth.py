"""
Assistente Jurídico SaaS IA V3
services/auth.py

Serviço de autenticação da plataforma.

Responsabilidades:

* autenticação de usuários;
* validação de credenciais;
* gerenciamento da sessão Streamlit;
* recuperação do usuário atual;
* logout;
* isolamento por organização;
* tratamento seguro de erros;
* compatibilidade com o app.py V3.
  """

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from db import get_connection
from security.passwords import verify_password

# ============================================================

# CONFIGURAÇÃO

# ============================================================

SESSION_USER_KEY = "user"

# ============================================================

# NORMALIZAÇÃO

# ============================================================

def _normalize_email(email: Any) -> str:
"""
Normaliza o e-mail antes da consulta.
"""
return str(email or "").strip().lower()

def _normalize_password(password: Any) -> str:
"""
Normaliza a senha sem alterar seu conteúdo interno.
"""
return str(password or "")

# ============================================================

# AUTENTICAÇÃO

# ============================================================

def authenticate(
email: str,
password: str,
) -> bool:
"""
Autentica um usuário.

```
Retorna:
    True  -> autenticação realizada.
    False -> credenciais inválidas.
"""

email = _normalize_email(email)
password = _normalize_password(password)

if not email or not password:
    return False

try:
    with get_connection() as c:

        row = c.execute(
            """
            SELECT
                id,
                organization_id,
                name,
                email,
                password_hash,
                role,
                created_at
            FROM users
            WHERE LOWER(email) = ?
            LIMIT 1
            """,
            (email,),
        ).fetchone()

    if not row:
        return False

    password_hash = row["password_hash"]

    if not password_hash:
        return False

    try:
        valid_password = verify_password(
            password,
            password_hash,
        )
    except Exception:
        return False

    if not valid_password:
        return False

    # Converte Row para dicionário.
    user = dict(row)

    # Nunca guardar o hash da senha na sessão.
    user.pop("password_hash", None)

    # Salva usuário autenticado.
    st.session_state[SESSION_USER_KEY] = user

    # Indicadores auxiliares.
    st.session_state["authenticated"] = True
    st.session_state["org_id"] = user.get(
        "organization_id"
    )

    return True

except Exception:
    return False
```

# ============================================================

# USUÁRIO ATUAL

# ============================================================

def get_current_user() -> Optional[Dict[str, Any]]:
"""
Retorna o usuário atualmente autenticado.

```
Retorna None quando não existe sessão válida.
"""

user = st.session_state.get(
    SESSION_USER_KEY
)

if not user:
    return None

if not isinstance(user, dict):
    return None

return user
```

# ============================================================

# STATUS DE AUTENTICAÇÃO

# ============================================================

def is_authenticated() -> bool:
"""
Verifica se existe usuário autenticado.
"""

```
user = get_current_user()

return bool(
    user
    and user.get("id")
    and user.get("organization_id")
)
```

# ============================================================

# ORGANIZAÇÃO ATUAL

# ============================================================

def get_current_org_id() -> Optional[int]:
"""
Retorna o ID da organização do usuário autenticado.
"""

```
user = get_current_user()

if not user:
    return None

try:
    return int(
        user["organization_id"]
    )
except (
    KeyError,
    TypeError,
    ValueError,
):
    return None
```

# ============================================================

# ID DO USUÁRIO

# ============================================================

def get_current_user_id() -> Optional[int]:
"""
Retorna o ID do usuário autenticado.
"""

```
user = get_current_user()

if not user:
    return None

try:
    return int(
        user["id"]
    )
except (
    KeyError,
    TypeError,
    ValueError,
):
    return None
```

# ============================================================

# LOGOUT

# ============================================================

def logout() -> None:
"""
Encerra a sessão atual.
"""

```
st.session_state.pop(
    SESSION_USER_KEY,
    None,
)

st.session_state.pop(
    "authenticated",
    None,
)

st.session_state.pop(
    "org_id",
    None,
)
```

# ============================================================

# LIMPEZA DA SESSÃO

# ============================================================

def clear_auth_session() -> None:
"""
Alias para limpeza da sessão de autenticação.
"""

```
logout()
```

# ============================================================

# CONTEXTO DE AUTENTICAÇÃO

# ============================================================

def get_auth_context() -> Dict[str, Any]:
"""
Retorna informações estruturadas da sessão atual.
"""

```
user = get_current_user()

if not user:
    return {
        "authenticated": False,
        "user": None,
        "user_id": None,
        "organization_id": None,
        "role": None,
    }

return {
    "authenticated": True,
    "user": user,
    "user_id": get_current_user_id(),
    "organization_id": get_current_org_id(),
    "role": user.get("role"),
}
```

# ============================================================

# TESTE DO MÓDULO

# ============================================================

def self_test() -> Dict[str, Any]:
"""
Teste estrutural do módulo.

```
Não executa autenticação real.
"""

required = [
    "authenticate",
    "get_current_user",
    "is_authenticated",
    "get_current_org_id",
    "get_current_user_id",
    "logout",
    "clear_auth_session",
    "get_auth_context",
]

missing = [
    name
    for name in required
    if name not in globals()
]

return {
    "module": "services.auth",
    "status": (
        "ok"
        if not missing
        else "error"
    ),
    "required_functions": required,
    "missing_functions": missing,
}
```

# ============================================================

# EXECUÇÃO DIRETA

# ============================================================

if **name** == "**main**":

```
result = self_test()

print("=" * 60)
print("AUTH.PY V3 - SELF TEST")
print("=" * 60)

print(
    f"Status: {result['status']}"
)

print(
    f"Funções obrigatórias: "
    f"{len(result['required_functions'])}"
)

print(
    f"Funções ausentes: "
    f"{result['missing_functions']}"
)

print("=" * 60)

