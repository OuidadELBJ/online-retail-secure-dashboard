import streamlit as st
from core.database import init_db
from core.security import init_session, login, logout

# Configuration globale de la page
st.set_page_config(page_title="Iowa Liquor Dashboard", layout="wide")

# Initialisation de l'infrastructure
init_db()
init_session()

# Logique de routage
if not st.session_state.authenticated:
    # Masquer le menu latéral par défaut si non connecté
    st.markdown("""
        <style>
            [data-testid="collapsedControl"] {display: none;}
            [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    st.title("Portail d'Authentification")
    st.write("Veuillez saisir vos identifiants pour accéder au tableau de bord.")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            if login(username, password):
                st.success("Connexion réussie.")
                st.rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")
else:
    # Interface pour l'utilisateur connecté
    st.sidebar.title("Informations Session")
    st.sidebar.write(f"Utilisateur : **{st.session_state.username}**")
    st.sidebar.write(f"Rôle : **{st.session_state.role}**")
    st.sidebar.button("Se déconnecter", on_click=logout)

    st.title("Accueil du Tableau de Bord")
    st.info("Utilisez le menu latéral pour naviguer vers les indicateurs de performance, l'IA ou l'administration.")