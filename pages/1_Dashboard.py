import streamlit as st
import pandas as pd
import plotly.express as px

from core.security import check_access
from core import kpis

check_access()

st.set_page_config(layout="wide")

st.title("Sales Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("data/online_retail_clean_data.csv")

st.markdown("""
<style>

.kpi-card {
    background: var(--secondary-background-color);
    padding: 22px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.15);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.kpi-title {
    font-size: 14px;
    opacity: 0.7;
}

.kpi-value {
    font-size: 30px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# KPIs
# ----------------------------

# ---------------- KPI ROW 1 ----------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Revenue</div>
        <div class="kpi-value">${kpis.total_revenue(df):,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Orders</div>
        <div class="kpi-value">{kpis.total_orders(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Customers</div>
        <div class="kpi-value">{kpis.total_customers(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Basket</div>
        <div class="kpi-value">${kpis.average_basket(df):.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# espace pour éviter chevauchement
st.markdown("<br>", unsafe_allow_html=True)


# ---------------- KPI ROW 2 ----------------

col5, col6, col7 = st.columns(3)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Items Sold</div>
        <div class="kpi-value">{kpis.total_products_sold(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Returning Customers</div>
        <div class="kpi-value">{kpis.returning_customers(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Top Country</div>
        <div class="kpi-value">{kpis.top_country(df)}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ----------------------------
# REVENUE BY MONTH
# ----------------------------

st.subheader("Revenue Over Time")

revenue_month = kpis.revenue_by_month(df)

fig = px.line(
    revenue_month,
    x="month",
    y="net_amount",
    color="year",
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# TOP PRODUCTS
# ----------------------------

st.subheader("Top Products")

top_products = kpis.top_products(df)

fig = px.bar(
    top_products,
    x="net_amount",
    y="Description",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FLOP PRODUCTS
# ----------------------------

st.subheader("Worst Performing Products")

flop_products = kpis.flop_products(df)

fig = px.bar(
    flop_products,
    x="net_amount",
    y="Description",
    orientation="h",
    color="net_amount"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# REVENUE BY COUNTRY
# ----------------------------

st.subheader("Revenue by Country")

country_rev = kpis.revenue_by_country(df)

fig = px.pie(
    country_rev,
    names="Country",
    values="net_amount"
)

st.plotly_chart(fig, use_container_width=True)