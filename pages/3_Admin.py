import streamlit as st
import pandas as pd
import os
from core.security import check_access, hash_password
from core.database import create_user, get_all_users_df

# 1. Activation du Pare-feu
check_access()

# 2. Vérification des droits d'administration
if st.session_state.role != "Admin":
    st.error("Privilèges insuffisants. Cette page est réservée aux administrateurs.")
    st.stop()

st.title("Panneau d'Administration")

# Section 1 : Création de compte
st.subheader("Création d'un nouvel utilisateur")
with st.form("create_user_form"):
    new_username = st.text_input("Identifiant")
    new_password = st.text_input("Mot de passe provisoire", type="password")
    new_role = st.selectbox("Attribution du rôle", ["User", "Admin"])
    submit_create = st.form_submit_button("Créer le profil")

    if submit_create and new_username and new_password:
        hashed_pwd = hash_password(new_password)
        if create_user(new_username, hashed_pwd, new_role):
            st.success(f"Le profil '{new_username}' a été créé avec le rôle {new_role}.")
        else:
            st.error("Échec : Cet identifiant est déjà utilisé.")

st.markdown("---")

# Section 2 : Consultation des utilisateurs
st.subheader("Annuaire des accès")
users_df = get_all_users_df()
st.dataframe(users_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 3 : Exportation des données brutes
st.subheader("Exportation des données métiers")
st.write("Téléchargement du dataset nettoyé par l'équipe Data.")

data_path = "data/clean_data.parquet"
if os.path.exists(data_path):
    try:
        # Chargement rapide du parquet pour l'export en CSV
        df_export = pd.read_parquet(data_path)
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Télécharger le fichier source (CSV)",
            data=csv_data,
            file_name="iowa_liquor_export.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier de données : {e}")
else:
    st.warning("Le fichier de données 'clean_data.parquet' n'a pas encore été généré par l'Ingénieur Data (Membre 1).")