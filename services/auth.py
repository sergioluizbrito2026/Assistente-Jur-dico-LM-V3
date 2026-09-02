import streamlit as st
from db import get_connection
from security.passwords import verify_password

def authenticate(email, password):
    with get_connection() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        st.session_state.user = dict(row)
        return True
    return False

def get_current_user():
    return st.session_state.get("user")

def logout():
    st.session_state.pop("user", None)
