from __future__ import annotations

"""
Assistente Jurídico SaaS IA V3.1
services/auth.py

Autenticação e controle de sessão do usuário.
"""

from typing import Any, Dict, Optional

import streamlit as st

from db import get_connection
from security.passwords import verify_password

# ============================================================

# CONFIGURAÇÃO

# ============================================================

SESSION_USER_KEY = "current_user"

# ============================================================

# NORMALIZAÇÃO

# ============================================================

def _normalize_email(email: Any) -> str:
"""Normaliza o endereço de e-mail."""
return str(email or "").strip().lower()

def _normalize_password(password: Any) -> str:
"""Converte a senha para string sem alterar seu conteúdo."""
return str(password or "")

# ============================================================

# AUTENTICAÇÃO

# ============================================================

def authenticate(
email: str,
password: str,
) -> bool:
"""
Autentica um usuário utilizando e-mail e senha.

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
    with get_connection() as conn:

        row = conn.execute(
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

    user = dict(row)

    # Nunca guardar o hash da senha na sessão.
    user.pop("password_hash", None)

    st.session_state[SESSION_USER_KEY] = user
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user.get("id")
    st.session_state["organization_id"] = user.get(
        "organization_id"
    )
    st.session_state["org_id"] = user.get(
        "organization_id"
    )
    st.session_state["perfil_nome"] = user.get(
        "name",
        "",
    )
    st.session_state["perfil_email"] = user.get(
        "email",
        "",
    )
    st.session_state["perfil_role"] = user.get(
        "role",
        "",
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

if not user:
    return False

return bool(
    st.session_state.get(
        "authenticated",
        False,
    )
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

organization_id = user.get(
    "organization_id"
)

if organization_id is None:
    organization_id = st.session_state.get(
        "organization_id"
    )

if organization_id is None:
    organization_id = st.session_state.get(
        "org_id"
    )

try:
    return int(organization_id)

except (
    TypeError,
    ValueError,
):
    return None
```

# Compatibilidade com versões anteriores.

def get_current_organization_id() -> Optional[int]:
"""
Alias compatível para obter a organização atual.
"""

```
return get_current_org_id()
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

user_id = user.get("id")

if user_id is None:
    user_id = st.session_state.get(
        "user_id"
    )

try:
    return int(user_id)

except (
    TypeError,
    ValueError,
):
    return None
```

# ============================================================

# PERFIL

# ============================================================

def get_current_user_name() -> str:
"""
Retorna o nome do usuário atual.
"""

```
user = get_current_user()

if not user:
    return ""

return str(
    user.get(
        "name",
        "",
    )
    or ""
)
```

def get_current_user_email() -> str:
"""
Retorna o e-mail do usuário atual.
"""

```
user = get_current_user()

if not user:
    return ""

return str(
    user.get(
        "email",
        "",
    )
    or ""
)
```

def get_current_user_role() -> str:
"""
Retorna o perfil/função do usuário atual.
"""

```
user = get_current_user()

if not user:
    return ""

return str(
    user.get(
        "role",
        "",
    )
    or ""
)
```

# ============================================================

# LOGOUT

# ============================================================

def logout() -> None:
"""
Encerra a sessão do usuário atual.
"""

```
keys_to_remove = [
    SESSION_USER_KEY,
    "authenticated",
    "user_id",
    "organization_id",
    "org_id",
    "perfil_nome",
    "perfil_email",
    "perfil_role",
]

for key in keys_to_remove:
    st.session_state.pop(
        key,
        None,
    )
```

# ============================================================

# LIMPEZA DE SESSÃO

# ============================================================

def clear_auth_session() -> None:
"""
Limpa completamente os dados de autenticação.
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

# PROTEÇÃO DE PÁGINA

# ============================================================

def require_authentication() -> bool:
"""
Verifica se existe autenticação válida.

```
Retorna:
    True  -> usuário autenticado.
    False -> usuário não autenticado.
"""

if is_authenticated():
    return True

st.warning(
    "É necessário realizar o login para acessar esta área."
)

return False
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

required_functions = [
    "authenticate",
    "get_current_user",
    "is_authenticated",
    "get_current_org_id",
    "get_current_organization_id",
    "get_current_user_id",
    "get_current_user_name",
    "get_current_user_email",
    "get_current_user_role",
    "logout",
    "clear_auth_session",
    "get_auth_context",
    "require_authentication",
]

missing_functions = [
    name
    for name in required_functions
    if name not in globals()
]

return {
    "module": "services.auth",
    "status": (
        "ok"
        if not missing_functions
        else "error"
    ),
    "required_functions": required_functions,
    "missing_functions": missing_functions,
}
```

# ============================================================

# EXECUÇÃO DIRETA

# ============================================================

if **name** == "**main**":

```
result = self_test()

print("=" * 60)
print("AUTH.PY V3.1 - SELF TEST")
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
```
