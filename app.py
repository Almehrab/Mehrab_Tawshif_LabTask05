import json
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
st.set_page_config(page_title="Business Location Explorer",layout="wide")
st.title("business Location Explorer")
st.write(" business locations using clustering and dimensionality reduction.")

@st.cache_data
def load_data(path="business_locations.geojson"):
    with open(path, "r", encoding="utf-8") as file:
        geojson_data = json.load(file)
    rows = []
    for feature in geojson_data["features"]:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) < 2:
            continue
        lon, lat = coordinates[:2]
        rows.append({**properties,"lon": lon,"lat": lat})
    return pd.DataFrame(rows)

df = load_data()
numeric_columns = ["Floor_Area_sqm","Daily_Foot_Traffic","Community_Impact_Score","Annual_Revenue_k"]
available_features = [
    column for column in numeric_columns
    if column in df.columns]
st.sidebar.header("1. Select Features")
selected_features = st.sidebar.multiselect(
    "Features to use in the clustering model",
    options=available_features,
    default=available_features)
st.sidebar.header("2. Clustering")
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["K-means", "DBSCAN"])

if algorithm == "K-means":
    number_of_clusters = st.sidebar.slider(
        "Number of clusters",min_value=2,max_value=10,value=4)

else:
    st.sidebar.caption(
        "`eps` controls how close standardized points must be "
        "to be considered neighbours."
    )
    eps = st.sidebar.slider( "Neighbourhood radius (eps)",min_value=0.1,max_value=3.0, value=0.7,step=0.1)
    min_samples = st.sidebar.slider("Minimum samples", min_value=2,max_value=20, value=5)

if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()
working_df = df.copy()
X = working_df[selected_features].copy()
for column in selected_features:
    X[column] = pd.to_numeric(X[column], errors="coerce")
valid_rows = X.notna().all(axis=1)
working_df = working_df.loc[valid_rows].copy()
X = X.loc[valid_rows].copy()
if len(working_df) == 0:
    st.error("No valid rows remain after cleaning the selected features.")
    st.stop()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
if algorithm == "K-means":
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
else:
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
labels = model.fit_predict(X_scaled)
working_df["cluster_number"] = labels
working_df["cluster"] = working_df["cluster_number"].astype(str)
working_df.loc[
    working_df["cluster_number"] == -1,
    "cluster"
] = "Noise"
cluster_labels_without_noise = set(labels)
cluster_labels_without_noise.discard(-1)
number_of_clusters_found = len(cluster_labels_without_noise)
number_of_noise_points = int((labels == -1).sum())
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric(
    "Locations",
    len(working_df)
)
metric_col2.metric(
    "Clusters found",
    number_of_clusters_found
)
metric_col3.metric(
    "Noise points",
    number_of_noise_points
)
non_noise_mask = labels != -1
non_noise_labels = labels[non_noise_mask]

if (
    number_of_clusters_found >= 2
    and len(non_noise_labels) > number_of_clusters_found
):
    score = silhouette_score(
        X_scaled[non_noise_mask],
        non_noise_labels
    )

    st.metric("Silhouette score",f"{score:.3f}",help=( "Higher values generally indicate more compact and ""well-separated clusters."))
else:
    st.info("A silhouette score requires at least two non-noise clusters.")

with st.expander("Look at data"):
    st.dataframe(
        working_df.head(200),
        use_container_width=True
    )

    st.write(f"{len(working_df)} valid locations loaded.")
    if "Neighborhood" in working_df.columns:
        st.write(
            f"{working_df['Neighborhood'].nunique()} neighborhoods found."
        )
pca = PCA(n_components=2)
pca_values = pca.fit_transform(X_scaled)
working_df["PC1"] = pca_values[:, 0]
working_df["PC2"] = pca_values[:, 1]

explained_variance = pca.explained_variance_ratio_
map_tab, reduction_tab, summary_tab = st.tabs([
    "Map",
    "Dimensionality Reduction",
    "Cluster Summary"
])
with map_tab:
    st.subheader("Cluster Map")
    hover_columns = [
        column for column in ["Neighborhood","Category","Subcategory",*selected_features]
if column in working_df.columns]

    map_figure = px.scatter_map(
        working_df,
        lat="lat",
        lon="lon",
        color="cluster",
        hover_name=(
            "Neighborhood"
            if "Neighborhood" in working_df.columns
            else None
        ),
        hover_data=hover_columns,
        zoom=9.5,
        height=650,
        map_style="carto-darkmatter",
        title=f"{algorithm} Business Location Clusters"
    )

    map_figure.update_traces(
        marker={
            "size": 10,
            "opacity": 0.85
        }
    )

    st.plotly_chart(
        map_figure,
        use_container_width=True
    )
with reduction_tab:
    st.subheader("PCA Projection")

    st.write(
        f"PC1 explains {explained_variance[0] * 100:.1f}% "
        f"of the variance and PC2 explains "
        f"{explained_variance[1] * 100:.1f}%."
    )

    pca_figure = px.scatter(
        working_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_name=(
            "Neighborhood"
            if "Neighborhood" in working_df.columns
            else None
        ),
        hover_data=hover_columns,
        title=f"PCA View of {algorithm} Clusters"
    )

    pca_figure.update_traces(
        marker={
            "size": 9,
            "opacity": 0.8
        }
    )

    st.plotly_chart(
        pca_figure,
        use_container_width=True
    )
with summary_tab:
    st.subheader("Cluster Profiles")

    summary = (
        working_df
        .groupby("cluster")[selected_features]
        .mean()
        .round(2)
    )

    counts = (
        working_df["cluster"]
        .value_counts()
        .rename("Location_Count")
    )

    summary = summary.join(counts)

    st.dataframe(
        summary,
        use_container_width=True
    )

    selected_summary_feature = st.selectbox(
        "Compare clusters using",
        selected_features
    )

    chart_data = (
        working_df
        .groupby("cluster", as_index=False)[selected_summary_feature]
        .mean()
    )

    summary_figure = px.bar(
        chart_data,
        x="cluster",
        y=selected_summary_feature,
        color="cluster",
        title=(
            f"Average {selected_summary_feature} "
            f"by Cluster"
        )
    )

    st.plotly_chart(
        summary_figure,
        use_container_width=True
    )