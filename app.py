import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation Dashboard")
st.caption("Unsupervised learning with K-Means clustering for customer analytics")

@st.cache_data
def load_default_data():
    return pd.read_csv("data/Mall_Customers.csv")

def clean_columns(df):
    rename = {}
    for c in df.columns:
        key = c.strip().lower().replace("_", " ")
        if key in ["customerid", "customer id"]:
            rename[c] = "CustomerID"
        elif key in ["gender", "genre"]:
            rename[c] = "Gender"
        elif key == "age":
            rename[c] = "Age"
        elif "annual income" in key:
            rename[c] = "Annual Income (k$)"
        elif "spending score" in key:
            rename[c] = "Spending Score (1-100)"
    return df.rename(columns=rename)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        df = clean_columns(pd.read_csv(uploaded))
        source = "Uploaded dataset"
    else:
        df = load_default_data()
        source = "Bundled Mall Customers dataset"

    required = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    df = df.dropna(subset=["Age", "Annual Income (k$)", "Spending Score (1-100)"]).copy()
    max_k = min(10, len(df) - 1)
    k = st.slider("Number of clusters (K)", 2, max_k, min(5, max_k))

st.write(f"**Data source:** {source}  |  **Customers:** {len(df):,}")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Avg age", f"{df['Age'].mean():.1f} yrs")
c3.metric("Avg income", f"${df['Annual Income (k$)'].mean():.1f}k")
c4.metric("Avg spending score", f"{df['Spending Score (1-100)'].mean():.1f}")

# EDA
st.subheader("1. Exploratory Data Analysis")
eda1, eda2 = st.columns(2)
with eda1:
    fig = px.histogram(df, x="Age", nbins=20, title="Customer age distribution")
    st.plotly_chart(fig, use_container_width=True)
with eda2:
    gender_counts = df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig = px.bar(gender_counts, x="Gender", y="Count", title="Customers by gender")
    st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)", size="Age", color="Gender",
                 hover_data=["CustomerID", "Age"], title="Income vs spending score")
st.plotly_chart(fig, use_container_width=True)

# K selection
st.subheader("2. Find the Optimal K")
features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X = scaler.fit_transform(features)

ks = list(range(2, max_k + 1))
inertias, silhouettes = [], []
for kk in ks:
    model = KMeans(n_clusters=kk, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels))

m1, m2 = st.columns(2)
with m1:
    elbow_df = pd.DataFrame({"K": ks, "Inertia": inertias})
    fig = px.line(elbow_df, x="K", y="Inertia", markers=True, title="Elbow method")
    st.plotly_chart(fig, use_container_width=True)
with m2:
    sil_df = pd.DataFrame({"K": ks, "Silhouette Score": silhouettes})
    fig = px.line(sil_df, x="K", y="Silhouette Score", markers=True, title="Silhouette score")
    st.plotly_chart(fig, use_container_width=True)

best_k = ks[int(np.argmax(silhouettes))]
st.info(f"Best K by silhouette score: **{best_k}**. The dashboard currently uses K = **{k}** from the sidebar.")

# Clustering
st.subheader("3. K-Means Customer Segmentation")
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

centers_original = scaler.inverse_transform(model.cluster_centers_)
centers = pd.DataFrame(centers_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
centers["Cluster"] = range(k)

fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)", color="Cluster",
                 hover_data=["CustomerID", "Gender", "Age"], title=f"Customer segments (K={k})")
fig.add_scatter(x=centers["Annual Income (k$)"], y=centers["Spending Score (1-100)"],
                mode="markers+text", text=[f"C{i}" for i in range(k)], textposition="top center",
                marker=dict(size=16, symbol="x"), name="Centroids")
st.plotly_chart(fig, use_container_width=True)

# Profiles
st.subheader("4. Segment Profiles & Business Insights")
profile = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income_k=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean")
).reset_index()
profile["Share_%"] = profile["Customers"] / len(df) * 100
profile["Avg_Age"] = profile["Avg_Age"].round(1)
profile["Avg_Income_k"] = profile["Avg_Income_k"].round(1)
profile["Avg_Spending"] = profile["Avg_Spending"].round(1)
profile["Share_%"] = profile["Share_%"].round(1)

st.dataframe(profile, use_container_width=True, hide_index=True)

# Automatic segment labels based on relative income/spending
income_med = df["Annual Income (k$)"].median()
spend_med = df["Spending Score (1-100)"].median()

insights = []
for _, row in profile.iterrows():
    high_income = row["Avg_Income_k"] >= income_med
    high_spend = row["Avg_Spending"] >= spend_med
    if high_income and high_spend:
        name, action = "High-value customers", "Prioritize loyalty rewards, premium offers and personalized campaigns."
    elif high_income and not high_spend:
        name, action = "High-income low-spending", "Use personalized offers, product education and re-engagement campaigns."
    elif not high_income and high_spend:
        name, action = "Young/high-spending opportunity", "Promote affordable bundles, memberships and frequent-purchase offers."
    else:
        name, action = "Low-engagement customers", "Use low-cost promotions and campaigns designed to increase visit frequency."
    insights.append({"Cluster": int(row["Cluster"]), "Segment": name, "Recommended strategy": action})

insight_df = pd.DataFrame(insights)
for item in insights:
    st.markdown(f"**Cluster {item['Cluster']} — {item['Segment']}**")
    st.write(item["Recommended strategy"])

# Segment chart
chart_df = profile.copy()
chart_df["Cluster"] = chart_df["Cluster"].astype(str)
fig = px.bar(chart_df, x="Cluster", y="Customers", title="Customers in each segment")
st.plotly_chart(fig, use_container_width=True)

# Customer explorer
st.subheader("5. Customer Explorer")
selected_cluster = st.selectbox("Filter by cluster", ["All"] + [str(i) for i in sorted(df["Cluster"].unique())])
view = df if selected_cluster == "All" else df[df["Cluster"] == int(selected_cluster)]
st.dataframe(view.sort_values("CustomerID"), use_container_width=True, hide_index=True)

# Downloads
st.subheader("6. Download Results")
result_csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download segmented customer data", result_csv, "customer_segments.csv", "text/csv")

st.caption("Model: K-Means | Features: Annual Income and Spending Score | Scaling: StandardScaler | Random state: 42")
