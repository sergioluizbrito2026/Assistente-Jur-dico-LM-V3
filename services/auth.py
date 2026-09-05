```python
"""
Assistente Jurídico SaaS IA V3.1
services/auth.py

Autenticação e controle de sessão do usuário.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from db import get_user_by_email


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SESSION_USER_KEY = "current_user"


# ============================================================
# USUÁRIO ATUAL
# ============================================================

def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Retorna o usuário atualmente autenticado.

    Retorna:
        dict com os dados do usuário ou None.
    """

    user = st.session_state.get(SESSION_USER_KEY)

    if not user:
        return None

    if not isinstance(user, dict):
        return None

    return user


# ============================================================
# AUTENTICAÇÃO
# ============================================================

def authenticate(
    email: str,
    password: str,
) -> bool:
    """
    Autentica um usuário utilizando e-mail e senha.

    Retorna:
        True  -> autenticação realizada.
        False -> credenciais inválidas.
    """

    email = str(email or "").strip().lower()
    password = str(password or "")

    if not email or not password:
        return False

    try:
        user = get_user_by_email(email)
    except Exception:
        return False

    if not user:
        return False

    password_hash = user.get("password_hash")

    if not password_hash:
        return False

    try:
        from security.passwords import verify_password

        valid = verify_password(
            password,
            password_hash,
        )

    except Exception:
        return False

    if not valid:
        return False

    st.session_state[SESSION_USER_KEY] = dict(user)
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user.get("id")
    st.session_state["organization_id"] = user.get(
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


# ============================================================
# LOGOUT
# ============================================================

def logout() -> None:
    """
    Encerra a sessão do usuário atual.
    """

    keys_to_remove = [
        SESSION_USER_KEY,
        "authenticated",
        "user_id",
        "organization_id",
        "perfil_nome",
        "perfil_email",
        "perfil_role",
    ]

    for key in keys_to_remove:
        st.session_state.pop(
            key,
            None,
        )


# ============================================================
# VERIFICAÇÃO DE AUTENTICAÇÃO
# ============================================================

def is_authenticated() -> bool:
    """
    Verifica se existe usuário autenticado.
    """

    user = get_current_user()

    return bool(
        user
        and st.session_state.get(
            "authenticated",
            False,
        )
    )


# ============================================================
# ORGANIZAÇÃO ATUAL
# ============================================================

def get_current_organization_id() -> Optional[int]:
    """
    Retorna o ID da organização do usuário atual.
    """

    user = get_current_user()

    if not user:
        return None

    organization_id = user.get(
        "organization_id"
    )

    if organization_id is None:
        return None

    try:
        return int(organization_id)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# ID DO USUÁRIO
# ============================================================

def get_current_user_id() -> Optional[int]:
    """
    Retorna o ID do usuário autenticado.
    """

    user = get_current_user()

    if not user:
        return None

    user_id = user.get("id")

    if user_id is None:
        return None

    try:
        return int(user_id)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# PERFIL
# ============================================================

def get_current_user_name() -> str:
    """
    Retorna o nome do usuário atual.
    """

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


def get_current_user_email() -> str:
    """
    Retorna o e-mail do usuário atual.
    """

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


def get_current_user_role() -> str:
    """
    Retorna o perfil/função do usuário atual.
    """

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


# ============================================================
# PROTEÇÃO DE PÁGINA
# ============================================================

def require_authentication() -> bool:
    """
    Verifica autenticação.

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


# ============================================================
# TESTE DO MÓDULO
# ============================================================

def self_test() -> Dict[str, Any]:
    """
    Teste estrutural do módulo.
    """

    required_functions = [
        "authenticate",
        "get_current_user",
        "logout",
        "is_authenticated",
        "get_current_organization_id",
        "get_current_user_id",
        "get_current_user_name",
        "get_current_user_email",
        "get_current_user_role",
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


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":
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
```
