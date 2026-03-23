#  Tableau de Bord Interactif - Iowa Liquor Sales

> **Projet Académique - 2ème Année Génie Logiciel** > **Encadrant :** Prof. Meryem FAKHOURI AMR  

Ce projet consiste en la conception et le développement d'un tableau de bord analytique interactif (Business Intelligence) basé sur le jeu de données public **Iowa Liquor Sales**. L'application intègre des fonctionnalités d'analyse de données, de visualisation en temps réel, de sécurité des accès (RBAC) et d'intelligence artificielle (Machine Learning).

---

##  Architecture Logicielle

Le projet adopte une architecture **"Streamlit Modulaire"** respectant le principe de Séparation des Préoccupations (Separation of Concerns). Cette structure garantit un code maintenable, sécurisé et évite le couplage fort entre la logique métier, l'accès aux données et l'interface utilisateur.

```text
Dashboard_Projet/
├── app.py                 # Routeur global et Pare-feu de l'application
├── requirements.txt       # Dépendances du projet
├── setup_admin.py         # Script d'initialisation de la BDD et du compte Admin
├── core/                  # Couche Métier et Accès aux Données (Backend)
│   ├── database.py        # DAO (Data Access Object) pour SQLite
│   ├── security.py        # Logique d'authentification et hachage (Bcrypt)
│   ├── kpis.py            # Fonctions de calculs analytiques (Pandas)
│   └── ml.py              # Logique des modèles prédictifs (Scikit-Learn)
├── data/                  # Couche Données
│   └── clean_data.parquet # Fichier final nettoyé (ETL)
└── pages/                 # Couche Présentation (Vues Streamlit protégées)
    ├── 1_Dashboard.py     # Interface des KPIs et Graphiques
    ├── 2_IA.py            # Interface des prédictions Machine Learning
    └── 3_Admin.py         # Panneau de contrôle administrateur

```

## Fonctionnalités Principales

* **Sécurité & Authentification :**
* Système de connexion sécurisé avec hachage des mots de passe (Bcrypt).
* Gestion des rôles (Role-Based Access Control) : Admin et User.
* Protection des routes : redirection automatique si la session est expirée ou invalide.


* **Analyse de Données (Business Intelligence) :**
* Suivi du Chiffre d'Affaires, volumes vendus, et marges bénéficiaires par zone géographique.
* Analyse du panier moyen et identification des tops vendeurs/produits.
* Graphiques interactifs (Plotly) avec filtres dynamiques (période, région, catégorie).


* **Intelligence Artificielle :**
* Modèles prédictifs pour anticiper les ventes futures.
* Segmentation (Clustering) pour identifier des comportements d'achat.


* **Administration :**
* Interface dédiée pour la création de comptes utilisateurs.
* Exportation des données brutes nettoyées (Format CSV).



##  Technologies Utilisées

* **Langage :** Python 3.9+
* **Frontend / Framework :** Streamlit
* **Backend & Base de données :** SQLite (via Context Managers)
* **Sécurité :** bcrypt (Hachage cryptographique)
* **Data Science :** Pandas, PyArrow (pour le format .parquet)
* **Data Visualisation :** Plotly
* **Machine Learning :** Scikit-Learn

##  Installation et Déploiement Local

**1. Cloner le dépôt et se placer dans le dossier**

```bash
git clone <url_du_repo>
cd Dashboard_Projet

```

**2. Créer un environnement virtuel et installer les dépendances**

```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt

```

**3. Initialiser la base de données et le compte Administrateur**
*Cette commande est à exécuter une seule fois pour générer le fichier users.db.*

```bash
python3 setup_admin.py

```

**4. Lancer l'application**

```bash
python3 -m streamlit run app.py

```

L'application sera accessible sur `http://localhost:8501`. Utilisez les identifiants générés à l'étape 3 pour vous connecter.

##  Équipe du Projet et Répartition

* **Ayoub JNIEH :** Data Engineer / ETL (Extraction, Nettoyage, Dictionnaire de données)
* **Ouidad EL BOJADDAINI :** Backend & Sécurité (Authentification, SQLite, RBAC, Architecture)
* **Ikram MOULAY :** Data Analyst / DataViz (Calcul des KPIs, Tableaux de bord Plotly)
* **Ghali BENYAHIA :** Ingénieur Machine Learning (Modèles prédictifs, Mise en cache, Déploiement)

```

```
