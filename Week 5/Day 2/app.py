import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ULTRA HIGH-CONTRAST & CAMERA-FRIENDLY CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* 🌌 High-Contrast Matte Deep-Space Background (Glare Resistant) */
    .stApp {
        background: #090d16 !important;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(79, 70, 229, 0.25) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.22) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
        color: #ffffff !important;
    }

    /* 🧭 Sharp High-Visibility Sidebar */
    [data-testid="stSidebar"] {
        background: #0d121f !important;
        border-right: 1.5px solid #2d3748 !important;
    }

    /* 🏷️ Sidebar Header Badge with Vivid Outline */
    .sidebar-header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1.5px solid #6366f1;
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
    }
    .sidebar-header-box h2 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 1.35rem;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .sidebar-header-box p {
        color: #38bdf8 !important;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }

    /* 🔘 Navigation Buttons with Clear Sharp State */
    div[data-testid="stRadio"] > div {
        gap: 0.45rem;
    }
    div[data-testid="stRadio"] label {
        background: #151c2e !important;
        border: 1.5px solid #2d3748 !important;
        padding: 0.65rem 1rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #312e81 !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
        transform: translateX(4px) !important;
    }

    /* 🚀 High-Luminance Hero Banner (Clear on Camera) */
    @keyframes bannerShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-banner {
        background: linear-gradient(135deg, #4338ca, #6d28d9, #0284c7);
        background-size: 200% 200%;
        animation: bannerShift 10s ease infinite;
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: #ffffff !important;
        margin-bottom: 2rem;
        border: 2px solid #a5b4fc;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-weight: 900;
        margin-bottom: 0.3rem;
        font-size: 2.3rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .hero-banner p {
        color: #f8fafc !important;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0;
    }

    /* 📦 High-Contrast Section Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #111827 !important;
        border: 1.5px solid #374151 !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
    }

    /* 💎 Vivid KPI Metric Cards (Crisp Camera Reading) */
    [data-testid="stMetric"] {
        background: #131b2e !important;
        border: 1.5px solid #3b82f6 !important;
        padding: 1.3rem !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        color: #93c5fd !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-weight: 900 !important;
        font-size: 2.1rem !important;
        color: #ffffff !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.8) !important;
    }

    /* 📢 Vivid Strategy Cards with High Edge Definition */
    .strategy-card {
        padding: 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.3rem;
    }
    .strat-premium {
        background: #1e1b18;
        border: 2px solid #eab308;
    }
    .strat-budget {
        background: #132219;
        border: 2px solid #22c55e;
    }
    .strat-risk {
        background: #251417;
        border: 2px solid #ef4444;
    }
    .strat-digital {
        background: #131c2e;
        border: 2px solid #3b82f6;
    }
    .strategy-card h3 {
        font-weight: 800 !important;
    }
    .strategy-card p, .strategy-card li {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* ⚡ Solid High-Vis Action Buttons */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 0.8rem 1.4rem !important;
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1.5px solid #60a5fa !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.5) !important;
    }
    .stDownloadButton > button:hover {
        background: #1d4ed8 !important;
        border-color: #93c5fd !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data
def load_data():
    base_path = r"D:\AI-Lab-99"
    dataset_path = None

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower() == "final dataset.csv":
                dataset_path = os.path.join(root, file)
                break
        if dataset_path is not None:
            break

    if dataset_path is None:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(".csv"):
                    if "final" in file.lower():
                        dataset_path = os.path.join(root, file)
                        break
            if dataset_path is not None:
                break

    if dataset_path is None:
        return None, None

    data = pd.read_csv(dataset_path)
    return data, dataset_path


df, dataset_path = load_data()


# ============================================================
# DATASET CHECK
# ============================================================

if df is None:
    st.error("❌ Final Dataset could not be found.")
    st.info("Make sure your final CSV dataset exists inside `D:\\AI-Lab-99`")
    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
        <div class="sidebar-header-box">
            <h2>👥 Analytics Pro</h2>
            <p>AI Customer Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home",
            "📊 Dashboard",
            "👥 Customer Segments",
            "🔎 Customer Search",
            "🤖 Customer Prediction",
            "📈 Visualizations",
            "📢 Marketing Strategy",
            "⬇️ Download"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    with st.container(border=True):
        st.markdown("<p style='color:#38bdf8; font-weight:800; margin:0;'>🎯 System Architecture</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#e2e8f0; font-size:0.88rem; margin-top:5px;'>• <b>Model:</b> K-Means Unsupervised<br>• <b>Engine:</b> Scikit-Learn & Streamlit</p>", unsafe_allow_html=True)

    st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>Customer Personality Analysis | ML Project</p>", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown("""
        <div class="hero-banner">
            <h1>👥 Customer Intelligence Platform</h1>
            <p>Interactive customer personality profiling, behavioral clustering & actionable marketing strategy portal.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("TOTAL CUSTOMERS", f"{len(df):,}")

    with col2:
        st.metric("TOTAL FEATURES", len(df.columns))

    with col3:
        segments = df["Customer_Segment"].nunique() if "Customer_Segment" in df.columns else 0
        st.metric("CUSTOMER SEGMENTS", segments)

    with col4:
        st.metric("SYSTEM HEALTH", "Active & Online")

    st.markdown("<h3 style='color:#ffffff; font-weight:800; margin-top:1.5rem;'>📌 Overview & Architecture</h3>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        with st.container(border=True):
            st.markdown("""
            <h4 style='color:#38bdf8; font-weight:800;'>💡 What this platform delivers:</h4>
            <ul style='color:#f8fafc; font-size:0.95rem; line-height:1.7;'>
                <li><b>Behavioral Profiling:</b> Advanced unsupervised clustering segmentation via K-Means.</li>
                <li><b>Dynamic Filtering:</b> Filter across income brackets, shopping channels, and recency tiers.</li>
                <li><b>Commercial Strategy:</b> Prescriptive action points designed for each unique consumer cohort.</li>
            </ul>
            """, unsafe_allow_html=True)
            st.success("✅ Connected to local database repository.")

    with c_right:
        with st.container(border=True):
            st.markdown("<h4 style='color:#38bdf8; font-weight:800;'>📁 Active File Source</h4>", unsafe_allow_html=True)
            st.code(dataset_path, language="plaintext")
            st.markdown("<p style='color:#cbd5e1; font-size:0.85rem;'>Path dynamically resolved through multi-level directory search.</p>", unsafe_allow_html=True)


# ============================================================
# DASHBOARD PAGE
# ============================================================

elif page == "📊 Dashboard":

    st.markdown("""
        <div class="hero-banner">
            <h1>📊 Executive Business Dashboard</h1>
            <p>Filter customer records in real-time to analyze cohort metrics and spending dynamics.</p>
        </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # FILTERS
    # ========================================================

    with st.container(border=True):
        st.markdown("<h4 style='color:#38bdf8; font-weight:800;'>🔎 Filter Criteria</h4>", unsafe_allow_html=True)
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            if "Customer_Segment" in df.columns:
                segment_options = ["All Segments"] + sorted(
                    df["Customer_Segment"].dropna().astype(str).unique().tolist()
                )
                selected_segment = st.selectbox("🏷️ Customer Segment", segment_options)
            else:
                selected_segment = "All Segments"

        with filter_col2:
            if "Customer_Activity_Level" in df.columns:
                activity_options = ["All Activity Levels"] + sorted(
                    df["Customer_Activity_Level"].dropna().astype(str).unique().tolist()
                )
                selected_activity = st.selectbox("⚡ Activity Level", activity_options)
            else:
                selected_activity = "All Activity Levels"

        with filter_col3:
            if "Preferred_Shopping_Channel" in df.columns:
                channel_options = ["All Channels"] + sorted(
                    df["Preferred_Shopping_Channel"].dropna().astype(str).unique().tolist()
                )
                selected_channel = st.selectbox("🛍️ Shopping Channel", channel_options)
            else:
                selected_channel = "All Channels"

        if "Income" in df.columns:
            valid_income = pd.to_numeric(df["Income"], errors="coerce").dropna()
            if len(valid_income) > 0:
                min_income = int(valid_income.min())
                max_income = int(valid_income.max())
                income_range = st.slider(
                    "💰 Income Range Filter",
                    min_income,
                    max_income,
                    (min_income, max_income)
                )
            else:
                income_range = (0, 0)
        else:
            income_range = (0, 0)

    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered_df = df.copy()

    if selected_segment != "All Segments" and "Customer_Segment" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Customer_Segment"].astype(str) == selected_segment]

    if selected_activity != "All Activity Levels" and "Customer_Activity_Level" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Customer_Activity_Level"].astype(str) == selected_activity]

    if selected_channel != "All Channels" and "Preferred_Shopping_Channel" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Preferred_Shopping_Channel"].astype(str) == selected_channel]

    if "Income" in filtered_df.columns:
        filtered_income = pd.to_numeric(filtered_df["Income"], errors="coerce")
        filtered_df = filtered_df[filtered_income.between(income_range[0], income_range[1])]

    # ========================================================
    # KPI CARDS
    # ========================================================

    st.markdown("<h3 style='color:#ffffff; font-weight:800; margin-top:1.5rem;'>📌 Cohort Performance Metrics</h3>", unsafe_allow_html=True)

    total_customers = len(filtered_df)
    average_income = pd.to_numeric(filtered_df["Income"], errors="coerce").mean() if (len(filtered_df) > 0 and "Income" in filtered_df.columns) else 0
    total_spending = pd.to_numeric(filtered_df["Total_spending"], errors="coerce").sum() if (len(filtered_df) > 0 and "Total_spending" in filtered_df.columns) else 0
    average_recency = pd.to_numeric(filtered_df["Recency"], errors="coerce").mean() if (len(filtered_df) > 0 and "Recency" in filtered_df.columns) else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("FILTERED USERS", f"{total_customers:,}")
    with k2:
        st.metric("AVERAGE INCOME", f"${average_income:,.0f}" if average_income > 0 else "N/A")
    with k3:
        st.metric("TOTAL SPEND", f"${total_spending:,.0f}" if total_spending > 0 else "N/A")
    with k4:
        st.metric("AVG RECENCY", f"{average_recency:.1f} days")

    # ========================================================
    # HIGH-VISIBILITY CHARTS
    # ========================================================

    st.markdown("<h3 style='color:#ffffff; font-weight:800; margin-top:1.5rem;'>📈 Visual Breakdown</h3>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown("<h5 style='color:#38bdf8; font-weight:800;'>👥 Customer Count by Segment</h5>", unsafe_allow_html=True)
            if "Customer_Segment" in filtered_df.columns and len(filtered_df) > 0:
                segment_chart = (
                    filtered_df["Customer_Segment"]
                    .value_counts()
                    .rename_axis("Segment")
                    .reset_index(name="Customers")
                )
                st.bar_chart(segment_chart.set_index("Segment"), color="#6366f1")
            else:
                st.info("No data to plot for segments.")

    with chart_col2:
        with st.container(border=True):
            st.markdown("<h5 style='color:#38bdf8; font-weight:800;'>🛒 Product Category Spending</h5>", unsafe_allow_html=True)
            spending_columns = [
                "MntWines", "MntFruits", "MntMeatProducts",
                "MntFishProducts", "MntSweetProducts", "MntGoldProds"
            ]
            available_spending = [col for col in spending_columns if col in filtered_df.columns]

            if len(available_spending) > 0 and len(filtered_df) > 0:
                spending_data = filtered_df[available_spending].sum().sort_values(ascending=False)
                st.bar_chart(spending_data, color="#06b6d4")
            else:
                st.info("No spending features available to plot.")

    # ========================================================
    # DATA TABLE
    # ========================================================

    st.markdown("<h3 style='color:#ffffff; font-weight:800; margin-top:1.5rem;'>📋 Filtered Records</h3>", unsafe_allow_html=True)

    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True, height=380)
    else:
        st.warning("⚠️ No customers match the selected filter criteria.")


# ============================================================
# CUSTOMER SEGMENTS PAGE
# ============================================================

elif page == "👥 Customer Segments":

    st.markdown("""
        <div class="hero-banner">
            <h1>👥 Cluster & Segment Breakdown</h1>
            <p>Detailed distribution of clustered consumer profiles generated by K-Means.</p>
        </div>
    """, unsafe_allow_html=True)

    if "Customer_Segment" in df.columns:
        segment_summary = df["Customer_Segment"].value_counts().reset_index()
        segment_summary.columns = ["Segment", "Customers"]
        segment_summary["Percentage"] = ((segment_summary["Customers"] / len(df)) * 100).round(2).astype(str) + "%"

        col_table, col_chart = st.columns([1.2, 1.8])

        with col_table:
            with st.container(border=True):
                st.markdown("<h4 style='color:#38bdf8; font-weight:800;'>📊 Segment Summary</h4>", unsafe_allow_html=True)
                st.dataframe(segment_summary, use_container_width=True, hide_index=True)

        with col_chart:
            with st.container(border=True):
                st.markdown("<h4 style='color:#38bdf8; font-weight:800;'>📈 Population Comparison</h4>", unsafe_allow_html=True)
                st.bar_chart(segment_summary.set_index("Segment")["Customers"], color="#a855f7")
    else:
        st.warning("⚠️ `Customer_Segment` column not found in the dataset.")


# ============================================================
# CUSTOMER SEARCH
# ============================================================

elif page == "🔎 Customer Search":

    st.markdown("""
        <div class="hero-banner">
            <h1>🔎 Customer Lookup Engine</h1>
            <p>Query any customer record across ID, demographics, or purchase behavior attributes.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        search_value = st.text_input("🔍 Search Database (Customer ID, Segment, Channel, etc.):", placeholder="Type to search...")

    if search_value:
        search_result = df[
            df.astype(str).apply(
                lambda row: row.str.contains(search_value, case=False, na=False).any(),
                axis=1
            )
        ]

        if len(search_result) > 0:
            st.success(f"🎯 Found **{len(search_result):,}** matching customer profile(s).")
            st.dataframe(search_result, use_container_width=True)
        else:
            st.warning("No records matched your search term.")
    else:
        st.info("💡 Enter an ID, name, or numerical value above to instantly filter customer profiles.")


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "🤖 Customer Prediction":

    st.markdown("""
        <div class="hero-banner">
            <h1>🤖 Real-Time Segment Predictor</h1>
            <p>Infer incoming customer clusters using pre-trained K-Means pipelines.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h3 style='color:#38bdf8; font-weight:800;'>⚙️ Model Status & Verification</h3>", unsafe_allow_html=True)
        st.info("The saved `kmeans_model.pkl` pipeline is staged for inference connection.")
        st.warning("⚠️ Feature sequencing validation is required before serving live predictions to ensure exact encoder compatibility.")


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":

    st.markdown("""
        <div class="hero-banner">
            <h1>📈 Deep-Dive Behavioral Visualizations</h1>
            <p>Analyze key distributions, revenue drivers, and product category patterns.</p>
        </div>
    """, unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        with st.container(border=True):
            if "Income" in df.columns:
                st.markdown("<h4 style='color:#38bdf8; font-weight:800;'>💰 Income Tier Distribution</h4>", unsafe_allow_html=True)
                st.bar_chart(df["Income"].value_counts(bins=10).sort_index(), color="#10b981")
            else:
                st.info("Income column not found.")

    with v2:
        with st.container(border=True):
            spending_columns = [
                "MntWines", "MntFruits", "MntMeatProducts",
                "MntFishProducts", "MntSweetProducts", "MntGoldProds"
            ]
            available_spending = [col for col in spending_columns if col in df.columns]

            if len(available_spending) > 0:
                st.markdown("<h4 style='color:#38bdf8; font-weight:800;'>🛒 Total Product Category Revenue</h4>", unsafe_allow_html=True)
                product_spending = df[available_spending].sum().sort_values(ascending=False)
                st.bar_chart(product_spending, color="#f59e0b")
            else:
                st.info("Spending columns not found.")


# ============================================================
# MARKETING STRATEGY
# ============================================================

elif page == "📢 Marketing Strategy":

    st.markdown("""
        <div class="hero-banner">
            <h1>📢 Prescriptive Marketing Action Plans</h1>
            <p>Tailored customer relationship management (CRM) tactics mapped directly to segments.</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="strategy-card strat-premium">
            <h3 style="margin-top: 0; color: #facc15;">👑 Premium / High-Value Customers</h3>
            <p>High purchasing power and consistent transactional recency.</p>
            <ul>
                <li>VIP concierge & early access product launches</li>
                <li>Tiered loyalty rewards & luxury product pairings</li>
                <li>Personalized quarterly gifting programs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="strategy-card strat-risk">
            <h3 style="margin-top: 0; color: #f87171;">⏱️ At-Risk & Churning Customers</h3>
            <p>High past value but high inactivity or recency lapse.</p>
            <ul>
                <li>Automated win-back discount sequences</li>
                <li>Feedback surveys with incentive rewards</li>
                <li>Direct re-engagement email outreach</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="strategy-card strat-budget">
            <h3 style="margin-top: 0; color: #4ade80;">🏷️ Budget & Value Seekers</h3>
            <p>Price-sensitive shoppers with seasonal transaction spikes.</p>
            <ul>
                <li>Discount bundles & bulk purchase promotions</li>
                <li>Coupon-based flash sales</li>
                <li>Affordable alternative product recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="strategy-card strat-digital">
            <h3 style="margin-top: 0; color: #60a5fa;">🌐 Digital-First Shoppers</h3>
            <p>High web-store engagement and email click-through rates.</p>
            <ul>
                <li>Targeted omni-channel push notifications</li>
                <li>Website homepage personalized carousels</li>
                <li>Exclusive app/online-only checkout perks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DOWNLOAD
# ============================================================

elif page == "⬇️ Download":

    st.markdown("""
        <div class="hero-banner">
            <h1>⬇️ Data Export Center</h1>
            <p>Export cleansed master datasets or cohort subsets directly in CSV format.</p>
        </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        with st.container(border=True):
            st.markdown("<h3 style='color:#38bdf8; font-weight:800;'>📦 Full Master Dataset</h3>", unsafe_allow_html=True)
            st.caption(f"Contains all {len(df):,} customer records with complete feature sets.")
            csv_full = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Master Dataset (CSV)",
                data=csv_full,
                file_name="Final_Dataset.csv",
                mime="text/csv",
                key="download_full"
            )

    with d2:
        with st.container(border=True):
            st.markdown("<h3 style='color:#38bdf8; font-weight:800;'>🎯 Filtered Dataset Subset</h3>", unsafe_allow_html=True)
            if "filtered_df" not in locals():
                filtered_df = df.copy()
            st.caption(f"Currently exporting {len(filtered_df):,} filtered records.")
            csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Filtered Subset (CSV)",
                data=csv_filtered,
                file_name="Filtered_Customer_Data.csv",
                mime="text/csv",
                key="download_filtered"
            )