import pandas as pd


# ----------------------------
# KPIs PRINCIPAUX
# ----------------------------

# chiffre d'affaires total
def total_revenue(df):
    return df["net_amount"].sum()


# nombre de commandes
def total_orders(df):
    return df["InvoiceNo"].nunique()


# nombre de clients
def total_customers(df):
    return df["CustomerID"].nunique()


# panier moyen
def average_basket(df):
    revenue = total_revenue(df)
    orders = total_orders(df)

    if orders == 0:
        return 0

    return revenue / orders


# ----------------------------
# KPIs BUSINESS
# ----------------------------

# produits vendus
def total_products_sold(df):
    return df["Quantity"].sum()


# nombre moyen d'articles par commande
def avg_items_per_order(df):
    items = df["Quantity"].sum()
    orders = total_orders(df)

    if orders == 0:
        return 0

    return items / orders


# clients récurrents
def returning_customers(df):
    customers = df.groupby("CustomerID")["InvoiceNo"].nunique()
    return (customers > 1).sum()


# ----------------------------
# TOP PERFORMANCES
# ----------------------------

# top produit
def top_product(df):
    return df.groupby("Description")["net_amount"].sum().idxmax()


# top pays
def top_country(df):
    return df.groupby("Country")["net_amount"].sum().idxmax()


# ----------------------------
# ANALYSES
# ----------------------------

# CA par mois
def revenue_by_month(df):
    return (
        df.groupby(["year", "month"])["net_amount"]
        .sum()
        .reset_index()
    )


# top produits
def top_products(df, n=10):
    return (
        df.groupby("Description")["net_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


# flops produits
def flop_products(df, n=10):
    return (
        df.groupby("Description")["net_amount"]
        .sum()
        .sort_values()
        .head(n)
        .reset_index()
    )


# ventes par pays
def revenue_by_country(df):
    return (
        df.groupby("Country")["net_amount"]
        .sum()
        .reset_index()
    )


# ventes par catégorie (si colonne existe)
def revenue_by_category(df):
    if "Category" not in df.columns:
        return None

    return (
        df.groupby("Category")["net_amount"]
        .sum()
        .reset_index()
    )