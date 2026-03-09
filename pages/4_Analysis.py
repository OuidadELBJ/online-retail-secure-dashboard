import streamlit as st
import pandas as pd
import plotly.express as px

from core.security import check_access
from core import kpis

check_access()
df = pd.read_csv("data/online_retail_clean_data.csv")

st.title("📈 Advanced Analysis")

# filtre pays
country = st.selectbox("Country", df["Country"].unique())

df_filtered = df[df["Country"] == country]


st.subheader("Flop Products")

flops = kpis.flop_products(df_filtered)

st.dataframe(flops)


st.subheader("Sales by Country")

country_sales = kpis.revenue_by_country(df)

fig = px.choropleth(
    country_sales,
    locations="Country",
    locationmode="country names",
    color="net_amount",
    color_continuous_scale="Blues",
    range_color=(0, country_sales["net_amount"].quantile(0.9)),
    hover_name="Country",
)

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(
        title="Revenue"
    )
)

st.plotly_chart(fig, use_container_width=True)