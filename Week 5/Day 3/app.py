import os, io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_PATH = Path(r"D:\AI-Lab-99")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#080b18 0%,#10142b 45%,#062a32 100%)}
[data-testid="stSidebar"]{background:#090d1b}
.hero{padding:28px 35px;border-radius:20px;background:linear-gradient(110deg,#5426d9,#087dcc);border:1px solid #7aa8ff;margin-bottom:25px}
.hero h1,.hero p{color:white}.hero h1{font-size:38px}
.kpi{padding:20px;border:1px solid #1683ff;border-radius:15px;background:#121b31;min-height:110px}
.kt{color:#65baff;font-size:14px;font-weight:700}.kv{color:white;font-size:30px;font-weight:800;margin-top:8px}
</style>
""", unsafe_allow_html=True)


# ---------------- FILES ----------------
def find_file(name):
    if not BASE_PATH.exists():
        return None
    for root, dirs, files in os.walk(BASE_PATH):
        dirs[:] = [d for d in dirs if d not in {".git","__pycache__",".venv","venv"}]
        for f in files:
            if f.lower() == name.lower():
                return Path(root) / f
    return None


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_pkl(path):
    return joblib.load(path)


# ---------------- MODEL HELPERS ----------------
def cluster_col(data):
    if data is None:
        return None
    for c in ["Cluster","cluster","Customer_Segment","Customer Segment","Segment","segment"]:
        if c in data.columns:
            return c
    return None


def model_feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)

    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None


def expected_features(model):
    n = getattr(model, "n_features_in_", None)
    if n is not None:
        return int(n)

    if hasattr(model, "named_steps"):
        for _, step in reversed(model.named_steps.items()):
            n = getattr(step, "n_features_in_", None)
            if n is not None:
                return int(n)
    return None


def clean_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()


def median_value(df, col):
    x = clean_num(df[col])
    return float(x.median()) if not x.empty else 0.0


def mode_value(df, col):
    x = df[col].dropna()
    if x.empty:
        return ""
    m = x.mode()
    return m.iloc[0] if len(m) else ""


def norm(x):
    return str(x).strip().lower().replace(" ","_")


def match_column(feature, columns):
    if feature in columns:
        return feature
    target = norm(feature)
    for c in columns:
        if norm(c) == target:
            return c
    return None


def safe_number_input(parent, label, df, col, key):
    s = clean_num(df[col])
    if s.empty:
        return 0.0

    lo, hi, med = float(s.min()), float(s.max()), float(s.median())

    if lo == hi:
        return parent.number_input(label, value=lo, format="%.4f", key=key)

    # FIX for StreamlitValueBelowMinError:
    # Never force min_value=0 because engineered features can have
    # negative values.
    default = float(np.clip(med, lo, hi))

    integer_cols = {
        "Recency","Age","Family_size","Family_Size",
        "NumWebPurchases","NumStorePurchases","NumCatalogPurchases",
        "NumDealsPurchases","NumWebVisitsMonth",
        "Kidhome","Teenhome","Children","Total_Purchase","Total_Purchases"
    }

    if col in integer_cols:
        ilo, ihi = int(np.floor(lo)), int(np.ceil(hi))
        default = int(np.clip(round(default), ilo, ihi))
        return parent.number_input(
            label, min_value=ilo, max_value=ihi,
            value=default, step=1, key=key
        )

    return parent.number_input(
        label, min_value=lo, max_value=hi,
        value=default, format="%.4f", key=key
    )


def build_prediction_row(inputs, df, model):
    names = model_feature_names(model)

    # BEST CASE: fitted pipeline/model remembers exact feature names.
    if names:
        row = {}
        for feature in names:
            source = match_column(feature, df.columns)

            if feature in inputs:
                row[feature] = inputs[feature]
            elif source in inputs if source else False:
                row[feature] = inputs[source]
            elif source:
                if pd.api.types.is_numeric_dtype(df[source]):
                    row[feature] = median_value(df, source)
                else:
                    row[feature] = mode_value(df, source)
            else:
                row[feature] = 0.0

        return pd.DataFrame([row], columns=names), False

    # FALLBACK: model does not expose feature names.
    # We exclude obvious ID/cluster columns and use a deterministic order.
    n = expected_features(model)
    if n is None:
        raise ValueError(
            "Saved model has neither feature_names_in_ nor n_features_in_. "
            "Exact inference cannot be verified."
        )

    cc = cluster_col(df)
    numeric = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != cc and not norm(c).startswith("unnamed")
        and norm(c) not in {"id","customer_id","customerid"}
    ]

    if len(numeric) < n:
        raise ValueError(
            f"Model expects {n} features, but only {len(numeric)} usable "
            f"numeric features are available in Final Dataset.csv."
        )

    selected = numeric[:n]
    values = []
    for c in selected:
        v = inputs[c] if c in inputs else median_value(df,c)
        values.append(0.0 if pd.isna(v) else float(v))

    return pd.DataFrame([values], columns=selected), True


def predict_customer(inputs, df, model):
    row, fallback = build_prediction_row(inputs, df, model)
    expected = expected_features(model)

    if expected is not None and row.shape[1] != expected:
        raise ValueError(
            f"Feature mismatch: model expects {expected}, "
            f"but prediction row contains {row.shape[1]}."
        )

    pred = model.predict(row)
    if len(pred) == 0:
        raise ValueError("Model returned no prediction.")

    return pred[0], row, fallback


# ---------------- LOAD ----------------
csv_path = find_file("Final Dataset.csv")
model_path = find_file("kmeans_model.pkl")
results_path = find_file("day3_results.pkl")

df = model = None
results = {}

if csv_path:
    try:
        df = load_csv(str(csv_path))
    except Exception as e:
        st.session_state["csv_error"] = str(e)

if model_path:
    try:
        model = load_pkl(str(model_path))
    except Exception as e:
        st.session_state["model_error"] = str(e)

if results_path:
    try:
        x = load_pkl(str(results_path))
        if isinstance(x, dict):
            results = x
        else:
            st.session_state["results_error"] = "day3_results.pkl is not a dictionary."
    except Exception as e:
        st.session_state["results_error"] = str(e)


# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    '<h2 style="color:white">👥 Analytics Pro</h2>'
    '<p style="color:#38bdf8">AI Customer Intelligence</p>',
    unsafe_allow_html=True
)

page = st.sidebar.radio("Navigation", [
    "🏠 Home","📊 Dashboard","👥 Customer Segments",
    "🔎 Customer Search","🤖 Customer Prediction",
    "📈 Visualizations","📢 Marketing Strategy","⬇️ Download"
])

st.sidebar.markdown("---")
st.sidebar.info("🎯 System Architecture\n\n• Model: K-Means Unsupervised\n• Engine: Scikit-Learn\n• Application: Streamlit")

if df is not None:
    st.sidebar.success(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")
else:
    st.sidebar.warning("Dataset not loaded")

if model is not None:
    st.sidebar.success("K-Means model loaded")
else:
    st.sidebar.warning("K-Means model not loaded")


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


# ---------------- HOME ----------------
if page == "🏠 Home":
    hero("👥 Customer Intelligence Platform",
         "Interactive customer personality profiling, behavioral clustering & actionable marketing strategy portal.")

    cc = cluster_col(df)
    values = [
        len(df) if df is not None else 0,
        len(df.columns) if df is not None else 0,
        int(df[cc].nunique()) if df is not None and cc else 0,
        "Active & Online" if df is not None and model is not None else "Needs Attention"
    ]

    cols = st.columns(4)
    for col, title, value in zip(
        cols,
        ["TOTAL CUSTOMERS","TOTAL FEATURES","CUSTOMER SEGMENTS","SYSTEM HEALTH"],
        values
    ):
        size = "30px" if isinstance(value,(int,float,np.integer,np.floating)) else "22px"
        display = f"{value:,}" if isinstance(value,(int,float,np.integer,np.floating)) else value
        col.markdown(
            f'<div class="kpi"><div class="kt">{title}</div>'
            f'<div class="kv" style="font-size:{size}">{display}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("## 📌 Overview & Architecture")
    a,b = st.columns(2)
    a.markdown("""
### 💡 What this platform delivers
- **Behavioral Profiling:** K-Means customer segmentation
- **Dynamic Filtering:** customer and segment analysis
- **Commercial Strategy:** marketing recommendations
- **Customer Prediction:** actual saved ML model
- **QA-aware inference:** feature count/order checked before prediction
""")
    b.markdown("### 📁 Active File Sources")
    for label,path in [
        ("Final Dataset.csv",csv_path),
        ("kmeans_model.pkl",model_path),
        ("day3_results.pkl",results_path)
    ]:
        if path:
            st.success(f"✅ {label}")
            st.code(str(path))
        else:
            st.warning(f"⚠️ {label} not found")


# ---------------- DASHBOARD ----------------
elif page == "📊 Dashboard":
    hero("📊 Customer Segmentation Dashboard",
         "Customer overview, segments and business intelligence.")

    if df is None:
        st.error("Final Dataset.csv not found.")
        st.stop()

    cc = cluster_col(df)
    cseries = df[cc] if cc else None

    a,b,c,d = st.columns(4)
    a.metric("Total Customers",f"{len(df):,}")
    b.metric("Total Features",len(df.columns))
    c.metric("Customer Segments",int(cseries.nunique()) if cseries is not None else "N/A")
    d.metric("Model Status","Loaded" if model is not None else "Not Found")

    st.divider()
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(25),use_container_width=True)

    if cseries is not None:
        st.subheader("👥 Segment Distribution")
        st.bar_chart(cseries.value_counts().sort_index())

    st.subheader("📊 Numeric Statistics")
    st.dataframe(df.select_dtypes(include=np.number).describe().T,use_container_width=True)


# ---------------- SEGMENTS ----------------
elif page == "👥 Customer Segments":
    hero("👥 Customer Segments","Explore customer groups and their characteristics.")

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    cc = cluster_col(df)

    if cc:
        values = sorted(df[cc].dropna().unique().tolist())
        seg = st.selectbox("Select Customer Segment",values)
        sd = df[df[cc] == seg].copy()

        a,b,c = st.columns(3)
        a.metric("Customers",len(sd))
        b.metric("Average Income",f"{sd['Income'].mean():,.2f}" if "Income" in sd else "N/A")

        spend = next((x for x in ["Total_spending","Total_Spending","TotalSpent"] if x in sd),None)
        c.metric("Average Spending",f"{sd[spend].mean():,.2f}" if spend else "N/A")

        st.dataframe(sd,use_container_width=True)

    else:
        st.warning("No cluster/segment column available.")


# ---------------- SEARCH ----------------
elif page == "🔎 Customer Search":
    hero("🔎 Customer Search","Search across the final dataset.")

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    q = st.text_input("Search customer",placeholder="Income, education, marital status, etc.")

    if q:
        mask = df.astype(str).apply(
            lambda s: s.str.contains(q,case=False,na=False,regex=False)
        ).any(axis=1)
        st.write(f"Found **{int(mask.sum())}** matching customers.")
        st.dataframe(df[mask],use_container_width=True)
    else:
        st.dataframe(df.head(50),use_container_width=True)


# ---------------- PREDICTION ----------------
elif page == "🤖 Customer Prediction":
    hero("🤖 Real-Time Segment Predictor",
         "Infer incoming customer clusters using the actual trained K-Means model.")

    with st.container(border=True):
        st.subheader("⚙️ Model Status & Verification")

        if model is not None:
            names = model_feature_names(model)
            n = expected_features(model)

            st.success("✅ Actual kmeans_model.pkl loaded successfully.")
            st.caption(str(model_path))

            if n is not None:
                st.info(f"Model expects **{n} features**.")

            if names:
                st.success(f"✅ {len(names)} exact model feature names detected.")
                st.caption("Prediction will follow the fitted model feature order.")
            else:
                st.warning(
                    "⚠️ feature_names_in_ was not stored in the model. "
                    "A deterministic numeric fallback will be used if necessary."
                )

            if hasattr(model,"named_steps"):
                st.caption("Pipeline: " + " → ".join(model.named_steps.keys()))

            st.caption(f"Final estimator: {type(list(model.named_steps.values())[-1]).__name__}"
                       if hasattr(model,"named_steps") else
                       f"Estimator: {type(model).__name__}")

        else:
            st.error("❌ kmeans_model.pkl was not found or could not be loaded.")
            if "model_error" in st.session_state:
                st.code(st.session_state["model_error"])

    if df is not None and model is not None:
        st.subheader("👤 Customer Input")
        inputs = {}

        input_defs = [
            ("Income","Income"),
            ("Recency","Recency"),
            ("Age","Age"),
            ("Total_spending","Total Spending"),
            ("Total_Purchase","Total Purchase"),
            ("Family_size","Family Size"),
            ("NumWebPurchases","Web Purchases"),
            ("NumStorePurchases","Store Purchases"),
            ("NumCatalogPurchases","Catalog Purchases"),
        ]

        existing = [(c,l) for c,l in input_defs if c in df.columns]
        cols = st.columns(3)

        for i,(column,label) in enumerate(existing):
            inputs[column] = safe_number_input(
                cols[i % 3],label,df,column,
                f"prediction_{column}"
            )

        st.divider()
        st.caption(
            "ℹ️ Features not shown above are automatically completed from "
            "Final Dataset.csv using median/mode values. This keeps the "
            "prediction row compatible with the saved model."
        )

        if st.button("🔮 Predict Customer Segment",type="primary",use_container_width=True):
            try:
                pred,row,fallback = predict_customer(inputs,df,model)
                st.session_state["latest_prediction"] = pred

                st.success(f"🎯 Predicted Customer Segment: **{pred}**")

                a,b,c = st.columns(3)
                a.metric("Prediction",str(pred))
                b.metric("Features Sent",row.shape[1])
                b.metric("Expected by Model",str(expected_features(model)) if expected_features(model) else "Unknown")

                if fallback:
                    st.warning(
                        "Fallback feature selection was used because the saved "
                        "model does not expose feature names. For production, "
                        "saving the original preprocessing pipeline is preferred."
                    )

                with st.expander("🔍 View prediction row"):
                    st.dataframe(row,use_container_width=True)

                cc = cluster_col(df)
                if cc:
                    sd = df[df[cc] == pred]
                    if not sd.empty:
                        a,b,c = st.columns(3)
                        a.metric("Customers in Segment",len(sd))
                        b.metric("Average Income",
                                 f"{sd['Income'].mean():,.2f}" if "Income" in sd else "N/A")
                        spend = next((x for x in ["Total_spending","Total_Spending","TotalSpent"] if x in sd),None)
                        c.metric("Average Spending",
                                 f"{sd[spend].mean():,.2f}" if spend else "N/A")

                st.subheader("📢 Marketing Recommendations")
                keys = [
                    ("marketing_message","Marketing Message"),
                    ("preferred_channel","Preferred Channel"),
                    ("personalized_offer","Personalized Offer"),
                    ("campaign_timing","Campaign Timing"),
                    ("discount_strategy","Discount Strategy"),
                    ("retention_strategy","Retention Strategy"),
                    ("primary_product","Primary Product"),
                    ("secondary_product","Secondary Product"),
                    ("cross_selling","Cross Selling"),
                    ("upselling","Upselling"),
                ]

                shown = False
                if isinstance(results,dict):
                    for key,title in keys:
                        if results.get(key) is not None:
                            st.write(f"**{title}:** {results[key]}")
                            shown = True

                if not shown:
                    st.info("No saved Day 3 recommendations found.")

            except Exception as e:
                st.error("❌ Prediction could not be completed.")
                st.warning("Model and dataset feature compatibility must be verified.")
                st.code(str(e))


# ---------------- VISUALIZATIONS ----------------
elif page == "📈 Visualizations":
    hero("📈 Customer Visualizations","Explore customer behavior.")

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    nums = df.select_dtypes(include=np.number).columns.tolist()

    if nums:
        feature = st.selectbox("Select Numeric Feature",nums)
        st.subheader(f"📊 Distribution: {feature}")
        st.bar_chart(df[feature].value_counts().sort_index().head(50))
        st.dataframe(df[feature].describe().to_frame(),use_container_width=True)
    else:
        st.warning("No numeric features available.")


# ---------------- MARKETING ----------------
elif page == "📢 Marketing Strategy":
    hero("📢 Marketing Strategy","Business recommendations from Day 3.")

    if not results:
        st.warning("day3_results.pkl not found or contains no saved results.")
        if "results_error" in st.session_state:
            st.code(st.session_state["results_error"])
    else:
        st.success("✅ Day 3 business insights loaded.")

        keys = [
            ("marketing_message","Marketing Message"),
            ("preferred_channel","Preferred Channel"),
            ("personalized_offer","Personalized Offer"),
            ("campaign_timing","Campaign Timing"),
            ("discount_strategy","Discount Strategy"),
            ("retention_strategy","Retention Strategy"),
            ("primary_product","Primary Product"),
            ("secondary_product","Secondary Product"),
            ("cross_selling","Cross Selling"),
            ("upselling","Upselling"),
        ]

        for key,title in keys:
            if results.get(key) is not None:
                st.markdown(f"### {title}")
                st.info(str(results[key]))

        table = results.get("marketing_recommendation")
        if isinstance(table,pd.DataFrame):
            st.subheader("📋 Recommendation Table")
            st.dataframe(table,use_container_width=True)


# ---------------- DOWNLOAD ----------------
elif page == "⬇️ Download":
    hero("⬇️ Download","Download project outputs.")

    if df is not None:
        st.download_button(
            "⬇️ Download Final Dataset",
            df.to_csv(index=False).encode("utf-8"),
            "Final_Dataset.csv",
            "text/csv"
        )

    if results_path and results:
        bio = io.BytesIO()
        joblib.dump(results,bio)
        st.download_button(
            "⬇️ Download Day 3 Results",
            bio.getvalue(),
            "day3_results.pkl",
            "application/octet-stream"
        )

    st.subheader("📁 Detected Files")
    for label,path in [
        ("Final Dataset.csv",csv_path),
        ("kmeans_model.pkl",model_path),
        ("day3_results.pkl",results_path)
    ]:
        if path:
            st.success(f"✅ {label}")
            st.code(str(path))
        else:
            st.error(f"❌ {label} not found")


st.sidebar.markdown("---")
st.sidebar.caption("Customer Personality Analysis | ML Project")
