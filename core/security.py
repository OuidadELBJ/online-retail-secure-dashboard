import bcrypt
import streamlit as st
from core.database import get_user_by_username

def hash_password(password: str) -> str:
    """Génère un hash Bcrypt pour un mot de passe en clair."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe par rapport à son hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def init_session():
    """Initialise les variables de session Streamlit."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None

def login(username, password):
    """Gère le processus d'authentification."""
    user = get_user_by_username(username)
    if user and verify_password(password, user['password_hash']):
        st.session_state.authenticated = True
        st.session_state.username = user['username']
        st.session_state.role = user['role']
        return True
    return False

def logout():
    """Gère la déconnexion."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()

def check_access():
    """
    Fonction Pare-feu à appeler au début de chaque page.
    Arrête l'exécution si l'utilisateur n'est pas connecté.
    """
    if not st.session_state.get("authenticated", False):
        st.warning("Accès refusé. Veuillez vous connecter via la page d'accueil.")
        st.stop()