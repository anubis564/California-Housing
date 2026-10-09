
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# ============================ CONFIG ======================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "my_california_housing_model.pkl"

# features
FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]


DATASET_MEDIAN = 179_700  
DATASET_CAP = 500_001  

CITIES = {
    "Custom location": None,
    "Los Angeles": (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
}

USD = "&#36;"  # keeps Streamlit from treating two "$" signs as LaTeX
# ==========================================================================


# ====================== Custom code from the notebook =====================

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# ==========================================================================

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

# ------------------------------- Styling ---------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        padding: 2rem 2.4rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    }
    .hero h1 {margin: 0; font-size: 2.3rem; color: white; padding: 0;}
    .hero p {margin: .5rem 0 0; opacity: .92; font-size: 1.05rem;}
    .result-card {
        background: linear-gradient(135deg, #059669 0%, #0ea5e9 100%);
        border-radius: 22px;
        padding: 2rem 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.3);
        margin-top: .5rem;
    }
    .result-card .label {font-size: .95rem; letter-spacing: .12em;
        text-transform: uppercase; opacity: .85;}
    .result-card .price {font-size: 3.2rem; font-weight: 800; line-height: 1.15;
        margin: .3rem 0;}
    .result-card .sub {font-size: .95rem; opacity: .9;}
    .placeholder-card {
        border: 2px dashed rgba(128, 128, 128, .4);
        border-radius: 22px;
        padding: 2.4rem 1.5rem;
        text-align: center;
        opacity: .75;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: .8rem 1rem;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, .35);
        color: white;
    }
    div.stButton > button:disabled {opacity: .45; transform: none; box-shadow: none;}
    section[data-testid="stSidebar"] h2 {font-size: 1.05rem; margin-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- Helpers ---------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = MODEL_FILE
    if not path.exists():
        candidates = sorted(MODELS_DIR.glob("*.pkl")) + sorted(MODELS_DIR.glob("*.joblib"))
        if not candidates:
            return None, None
        path = candidates[0]
    return joblib.load(path), path


def model_display_name(model) -> str:
    # The saved model is a Pipeline: show the final estimator's name.
    if hasattr(model, "steps"):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__


def predict_price(model, row: dict) -> float:
    # The pipeline does all preprocessing (imputing, ratios, logs, clusters,
    # one-hot, scaling) itself, so it just needs the raw columns.
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return float(np.ravel(model.predict(X))[0])


def apply_city():
    coords = CITIES[st.session_state["city"]]
    if coords:
        st.session_state["lat"], st.session_state["lon"] = coords


# ------------------------------ Load model -------------------------------
try:
    model, model_path = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model: {e}")
    st.caption(
        "Run the app in the same environment you used in the notebook "
        "(same scikit-learn, pandas and numpy versions)."
    )
    st.stop()

if model is None:
    st.error(
        f"No model found. Put your model file in `{MODELS_DIR}` "
        "(expected `my_california_housing_model.pkl`)."
    )
    st.stop()

# ------------------------------- Header ----------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏡 California House Price Predictor</h1>
        <p>Describe a neighborhood block and its homes, and the model estimates the
        median house value for that block.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Sidebar ---------------------------------
st.session_state.setdefault("lat", 34.05)
st.session_state.setdefault("lon", -118.24)

with st.sidebar:
    st.title("Property features")
    st.caption("Adjust the values, then hit **Predict**.")

    st.markdown("## 📍 Location")
    st.selectbox("Quick pick a city", list(CITIES), key="city", on_change=apply_city)
    latitude = st.slider("Latitude", 32.5, 42.0, step=0.01, key="lat")
    longitude = st.slider("Longitude", -124.4, -114.3, step=0.01, key="lon")
    ocean_proximity = st.selectbox(
        "Ocean proximity", OCEAN_CATEGORIES, index=0,
        help="How close the block is to the ocean or bay. ISLAND is very rare in the "
        "training data (5 blocks), so predictions for it are less reliable.",
    )

    st.markdown("## 👥 Neighborhood")
    median_income = st.slider(
        "Median income (× 10,000 USD)", 0.5, 15.0, 3.5, 0.1,
        help="Median household income in the block, in tens of thousands of dollars.",
    )
    population = st.number_input(
        "Population", min_value=3, max_value=40_000, value=1_166, step=50,
        help="Total residents in the block.",
    )
    households = st.number_input(
        "Households", min_value=1, max_value=7_000, value=409, step=10,
        help="Total number of households in the block.",
    )

    st.markdown("## 🏠 Housing")
    housing_median_age = st.slider("Median house age (years)", 1, 52, 29)
    total_rooms = st.number_input(
        "Total rooms", min_value=1, max_value=40_000, value=2_127, step=50,
        help="Total rooms across all homes in the block.",
    )
    total_bedrooms = st.number_input(
        "Total bedrooms", min_value=1, max_value=7_000, value=435, step=10,
        help="Total bedrooms across all homes in the block.",
    )

    invalid_inputs = total_bedrooms > total_rooms
    if invalid_inputs:
        st.warning("Total bedrooms can't exceed total rooms.")

    st.divider()
    st.caption(f"Model: **{model_display_name(model)}**  \nFile: `{model_path.name}`")

row = {
    "longitude": float(longitude),
    "latitude": float(latitude),
    "housing_median_age": float(housing_median_age),
    "total_rooms": float(total_rooms),
    "total_bedrooms": float(total_bedrooms),
    "population": float(population),
    "households": float(households),
    "median_income": float(median_income),
    "ocean_proximity": ocean_proximity,
}

# ------------------------------ Main layout ------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.subheader("Location on the map")
    st.map(
        pd.DataFrame({"lat": [latitude], "lon": [longitude]}),
        zoom=6,
        size=1500,
        color="#ef4444",
    )

    
    m1, m2, m3 = st.columns(3)
    m1.metric("Rooms per house", f"{total_rooms / households:.1f}")
    m2.metric("Bedrooms ratio", f"{total_bedrooms / total_rooms:.2f}")
    m3.metric("People per house", f"{population / households:.1f}")

    with st.expander("Review your inputs"):
        st.dataframe(pd.DataFrame(row, index=["Value"]).T.astype(str))

with right:
    st.subheader("Prediction")
    predict_clicked = st.button(
        "✨ Predict house price", type="primary", disabled=invalid_inputs
    )

    if predict_clicked:
        try:
            price = predict_price(model, row)
        except Exception as e:  
            st.error(f"Prediction failed: {e}")
        else:
            delta_pct = (price - DATASET_MEDIAN) / DATASET_MEDIAN * 100
            direction = "above" if delta_pct >= 0 else "below"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Estimated median house value</div>
                    <div class="price">{USD}{price:,.0f}</div>
                    <div class="sub">{abs(delta_pct):.0f}% {direction} the California median
                    ({USD}{DATASET_MEDIAN:,.0f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if price >= DATASET_CAP * 0.97:
                st.info(
                    "The training data caps prices at about 500,000 USD, so estimates near "
                    "that level should be read as 'at least this expensive'."
                )
    else:
        st.markdown(
            """
            <div class="placeholder-card">
                <div style="font-size:2.4rem">🔮</div>
                <div>Your prediction will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    "Estimates come from a machine-learning model trained on 1990 California census data "
    "and are for demonstration only, not for real valuations."
)