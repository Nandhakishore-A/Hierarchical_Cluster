import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Iris Hierarchical Clustering", layout="wide")
st.title("🌸 Iris Hierarchical Clustering App")

# -----------------------------
# LOAD TRAINED OBJECTS
# -----------------------------
@st.cache_resource
def load_objects():
    hc_model = joblib.load("hierarchical_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("features.pkl")
    return hc_model, scaler, le, feature_cols

hc_model, scaler, le, FEATURE_COLUMNS = load_objects()

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("DATA_SETS/iris_data.csv")

# Validate RAW columns only
REQUIRED_COLUMNS = [
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width",
    "Species"
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"❌ Dataset mismatch. Missing columns: {missing}")
    st.stop()

# -----------------------------
# ENCODE + SCALE + CLUSTER
# -----------------------------
df["Species_encoded"] = le.transform(df["Species"])

X = df[FEATURE_COLUMNS]
X_scaled = scaler.transform(X)

df["Cluster"] = hc_model.fit_predict(X_scaled)

# -----------------------------
# MAIN DATA VIEW
# -----------------------------
st.subheader("📄 Clustered Dataset (with Encoding)")
st.dataframe(
    df[[
        "Species",
        "Species_encoded",
        "Cluster",
        "Sepal.Length",
        "Sepal.Width",
        "Petal.Length",
        "Petal.Width"
    ]],
    use_container_width=True
)

# -----------------------------
# SPECIES → ENCODED → CLUSTER MAPPING
# -----------------------------
st.subheader("🧬 Species Encoding & Cluster Mapping")

mapping_df = (
    df[["Species", "Species_encoded", "Cluster"]]
    .drop_duplicates()
    .sort_values(["Species_encoded", "Cluster"])
    .reset_index(drop=True)
)

st.table(mapping_df)

# -----------------------------
# CLUSTER DISTRIBUTION
# -----------------------------
st.subheader("📊 Cluster Distribution")
st.write(df["Cluster"].value_counts().sort_index())

# -----------------------------
# VISUALIZATION
# -----------------------------
st.subheader("📈 Cluster Visualization")

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Sepal.Length",
    y="Petal.Length",
    hue="Cluster",
    palette="viridis",
    ax=ax
)

ax.set_title("Hierarchical Clustering Result")
st.pyplot(fig)

# -----------------------------
# PREDICT NEW SAMPLE
# -----------------------------
st.subheader("🔮 Predict Cluster (New Sample)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
    petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)

with col2:
    petal_width = st.number_input("Petal Width", 0.1, 2.5, 0.2)
    species = st.selectbox("Species", le.classes_)

if st.button("Predict Cluster"):
    species_encoded = le.transform([species])[0]

    input_df = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            species_encoded
        ]],
        columns=FEATURE_COLUMNS
    )

    input_scaled = scaler.transform(input_df)
    combined = np.vstack([X_scaled, input_scaled])

    predicted_cluster = hc_model.fit_predict(combined)[-1]

    st.success("✅ Prediction Result")
    st.write(f"**Species:** {species}")
    st.write(f"**Species Encoded:** {species_encoded}")
    st.write(f"**Predicted Cluster:** {predicted_cluster}")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("ℹ️ Info")
st.sidebar.markdown("""
**Algorithm:** Agglomerative Clustering  
**Encoding:** LabelEncoder  
**Scaling:** StandardScaler  

✔ Species → Encoded shown  
✔ Cluster mapping shown  
✔ No dataset mismatch  
✔ No fcluster  
""")
