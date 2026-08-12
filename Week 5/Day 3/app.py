import os
from pathlib import Path
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title='Customer Intelligence Platform', page_icon='👥', layout='wide', initial_sidebar_state='expanded')
BASE_PATH = Path(r'D:\AI-Lab-99')

st.markdown('''<style>
.stApp{background:linear-gradient(135deg,#080b18 0%,#10142b 45%,#062a32 100%)}
[data-testid="stSidebar"]{background:#090d1b}
.hero{padding:28px 35px;border-radius:20px;background:linear-gradient(110deg,#5426d9,#087dcc);border:1px solid #7aa8ff;margin-bottom:25px}
.hero h1,.hero p{color:white}.hero h1{font-size:38px}
.kpi{padding:20px;border:1px solid #1683ff;border-radius:15px;background:#121b31;min-height:110px}.kt{color:#65baff;font-size:14px;font-weight:700}.kv{color:white;font-size:30px;font-weight:800;margin-top:8px}
</style>''', unsafe_allow_html=True)

def find_file(name):
    if not BASE_PATH.exists(): return None
    for root, dirs, files in os.walk(BASE_PATH):
        for f in files:
            if f.lower() == name.lower(): return Path(root) / f
    return None

@st.cache_data
def load_csv(path): return pd.read_csv(path)

@st.cache_resource
def load_pkl(path): return joblib.load(path)

def cluster_col(data):
    for c in ['Cluster','cluster','Customer_Segment','Customer Segment']:
        if c in data.columns: return c
    return None

def model_features(model):
    x=getattr(model,'feature_names_in_',None)
    if x is not None: return list(x)
    if hasattr(model,'named_steps'):
        for _,step in reversed(model.named_steps.items()):
            x=getattr(step,'feature_names_in_',None)
            if x is not None: return list(x)
    return None

def expected_features(model):
    x=getattr(model,'n_features_in_',None)
    if x is not None: return int(x)
    if hasattr(model,'named_steps'):
        for _,step in reversed(model.named_steps.items()):
            x=getattr(step,'n_features_in_',None)
            if x is not None: return int(x)
    return None

def predict_customer(inputs, df, model):
    names=model_features(model)
    if names:
        row={}
        for c in names:
            if c in inputs: row[c]=inputs[c]
            elif c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]): row[c]=df[c].median()
                else:
                    m=df[c].mode(); row[c]=m.iloc[0] if len(m) else ''
            else: row[c]=0
        return int(model.predict(pd.DataFrame([row]))[0])
    n=expected_features(model)
    nums=df.select_dtypes(include=np.number).columns.tolist()
    if n is None:
        n=len(nums)
    if len(nums)<n:
        raise ValueError(
            f'Model expects {n} features, but the final dataset exposes only '
            f'{len(nums)} numeric features. The original training/preprocessing '
            f'pipeline is required for exact inference.'
        )

    selected = nums[:n]
    vals=[]
    for c in selected:
        if c in inputs:
            value = inputs[c]
        else:
            value = pd.to_numeric(df[c], errors='coerce').median()
        if pd.isna(value):
            value = 0
        vals.append(float(value))

    arr=np.asarray(vals, dtype=float).reshape(1, -1)
    return int(model.predict(arr)[0])

# Files
csv_path=find_file('Final Dataset.csv')
model_path=find_file('kmeans_model.pkl')
results_path=find_file('day3_results.pkl')

df=None; model=None; results={}
if csv_path:
    try: df=load_csv(str(csv_path))
    except Exception as e: st.error(f'Dataset error: {e}')
if model_path:
    try: model=load_pkl(str(model_path))
    except Exception as e: st.session_state['model_error']=str(e)
if results_path:
    try: results=load_pkl(str(results_path))
    except Exception: results={}

st.sidebar.markdown('<h2 style="color:white">👥 Analytics Pro</h2><p style="color:#38bdf8">AI Customer Intelligence</p>',unsafe_allow_html=True)
page=st.sidebar.radio('Navigation',['🏠 Home','📊 Dashboard','👥 Customer Segments','🔎 Customer Search','🤖 Customer Prediction','📈 Visualizations','📢 Marketing Strategy','⬇️ Download'])
st.sidebar.markdown('---')
st.sidebar.info('🎯 System Architecture\n\n• Model: K-Means Unsupervised\n• Engine: Scikit-Learn\n• Application: Streamlit')

# HOME
if page=='🏠 Home':
    st.markdown('<div class="hero"><h1>👥 Customer Intelligence Platform</h1><p>Interactive customer personality profiling, behavioral clustering & actionable marketing strategy portal.</p></div>',unsafe_allow_html=True)
    vals=[len(df) if df is not None else 0,len(df.columns) if df is not None else 0,df[cluster_col(df)].nunique() if df is not None and cluster_col(df) else 2,'Active & Online' if df is not None else 'Dataset Missing']
    cols=st.columns(4)
    for col,title,val in zip(cols,['TOTAL CUSTOMERS','TOTAL FEATURES','CUSTOMER SEGMENTS','SYSTEM HEALTH'],vals):
        col.markdown(f'<div class="kpi"><div class="kt">{title}</div><div class="kv">{val:,}</div></div>',unsafe_allow_html=True) if isinstance(val,(int,float,np.integer,np.floating)) else col.markdown(f'<div class="kpi"><div class="kt">{title}</div><div class="kv" style="font-size:22px">{val}</div></div>',unsafe_allow_html=True)
    st.markdown('## 📌 Overview & Architecture')
    a,b=st.columns(2)
    a.markdown('''### 💡 What this platform delivers\n- **Behavioral Profiling:** K-Means customer segmentation\n- **Dynamic Filtering:** customer and segment analysis\n- **Commercial Strategy:** marketing recommendations\n- **Customer Prediction:** actual saved ML model''')
    b.markdown('### 📁 Active File Sources')
    for label,path in [('Final Dataset.csv',csv_path),('kmeans_model.pkl',model_path),('day3_results.pkl',results_path)]:
        if path: st.success(f'✅ {label} found'); st.code(str(path))
        else: st.warning(f'⚠️ {label} not found')

# DASHBOARD
elif page=='📊 Dashboard':
    st.markdown('<div class="hero"><h1>📊 Customer Segmentation Dashboard</h1><p>Customer overview, segments and business intelligence.</p></div>',unsafe_allow_html=True)
    if df is None: st.error('Final Dataset.csv not found.'); st.stop()
    c=df[cluster_col(df)] if cluster_col(df) else None
    a,b,c1,d=st.columns(4); a.metric('Total Customers',f'{len(df):,}'); b.metric('Total Features',len(df.columns)); c1.metric('Customer Segments',int(c.nunique()) if c is not None else 2); d.metric('Model Status','Loaded' if model is not None else 'Not Found')
    st.divider(); st.subheader('📋 Dataset Preview'); st.dataframe(df.head(25),use_container_width=True)
    if c is not None: st.subheader('👥 Segment Distribution'); st.bar_chart(c.value_counts().sort_index())
    st.subheader('📊 Numeric Statistics'); st.dataframe(df.select_dtypes(include=np.number).describe().T,use_container_width=True)

# SEGMENTS
elif page=='👥 Customer Segments':
    st.markdown('<div class="hero"><h1>👥 Customer Segments</h1><p>Explore customer groups and their characteristics.</p></div>',unsafe_allow_html=True)
    if df is None: st.error('Dataset not found.'); st.stop()
    cc=cluster_col(df)
    if cc:
        seg=st.selectbox('Select Customer Segment',sorted(df[cc].dropna().unique().tolist())); sd=df[df[cc]==seg].copy()
        x,y,z=st.columns(3); x.metric('Customers',len(sd)); y.metric('Average Income',f"{sd['Income'].mean():,.2f}" if 'Income' in sd else 'N/A'); z.metric('Average Spending',f"{sd['Total_spending'].mean():,.2f}" if 'Total_spending' in sd else 'N/A')
        st.dataframe(sd,use_container_width=True)
    elif isinstance(results.get('segment_customers'),pd.DataFrame): st.dataframe(results['segment_customers'],use_container_width=True)
    else: st.warning('No segment column/data available.')

# SEARCH
elif page=='🔎 Customer Search':
    st.markdown('<div class="hero"><h1>🔎 Customer Search</h1><p>Search across the final dataset.</p></div>',unsafe_allow_html=True)
    if df is None: st.stop()
    q=st.text_input('Search customer',placeholder='Enter income, education, marital status, etc.')
    if q:
        mask=df.astype(str).apply(lambda s:s.str.contains(q,case=False,na=False)).any(axis=1); st.write(f'Found **{int(mask.sum())}** matching customers.'); st.dataframe(df[mask],use_container_width=True)
    else: st.dataframe(df.head(50),use_container_width=True)

# PREDICTION / DAY 3
elif page=='🤖 Customer Prediction':
    st.markdown('<div class="hero"><h1>🤖 Real-Time Segment Predictor</h1><p>Infer incoming customer clusters using the actual trained K-Means model.</p></div>',unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader('⚙️ Model Status & Verification')
        if model is not None:
            st.success('✅ Actual kmeans_model.pkl loaded successfully.')
            st.caption(str(model_path)); n=expected_features(model); names=model_features(model)
            if n: st.info(f'Model expects {n} features. Inputs are completed from the final dataset when a training feature is not manually entered.')
            if names: st.caption(f'{len(names)} model feature names detected automatically.')
        else:
            st.error('❌ kmeans_model.pkl was not found or could not be loaded.')
            if 'model_error' in st.session_state: st.code(st.session_state['model_error'])
    if df is not None and model is not None:
        st.subheader('👤 Customer Input'); inputs={}; a,b,c=st.columns(3)
        # Safe input helper:
        # The previous version forced min_value=0, which crashes when a
        # dataset feature has a negative median. We now use the real
        # finite dataset range and always clamp the default inside it.
        def safe_number_input(key, label, col, integer=False):
            if key not in df.columns:
                return

            series = pd.to_numeric(df[key], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                return

            actual_min = float(series.min())
            actual_max = float(series.max())
            med = float(series.median())

            if actual_min == actual_max:
                if integer:
                    inputs[key] = col.number_input(label, value=int(round(actual_min)), step=1)
                else:
                    inputs[key] = col.number_input(label, value=float(actual_min), format="%.4f")
                return

            # Keep a little precision for continuous features.
            if integer:
                lo = int(np.floor(actual_min))
                hi = int(np.ceil(actual_max))
                default = int(np.clip(round(med), lo, hi))
                inputs[key] = col.number_input(
                    label,
                    min_value=lo,
                    max_value=hi,
                    value=default,
                    step=1
                )
            else:
                lo = float(actual_min)
                hi = float(actual_max)
                default = float(np.clip(med, lo, hi))
                inputs[key] = col.number_input(
                    label,
                    min_value=lo,
                    max_value=hi,
                    value=default,
                    format="%.4f"
                )

        safe_number_input('Income', 'Income', a, integer=False)
        safe_number_input('Recency', 'Recency', a, integer=True)
        safe_number_input('Age', 'Age', a, integer=True)

        safe_number_input('Total_spending', 'Total Spending', b, integer=False)
        safe_number_input('Total_Purchase', 'Total Purchase', b, integer=True)
        safe_number_input('Family_size', 'Family Size', b, integer=True)

        safe_number_input('NumWebPurchases', 'Web Purchases', c, integer=True)
        safe_number_input('NumStorePurchases', 'Store Purchases', c, integer=True)
        safe_number_input('NumCatalogPurchases', 'Catalog Purchases', c, integer=True)
        if st.button('🔮 Predict Customer Segment',type='primary'):
            try:
                pred=predict_customer(inputs,df,model); st.session_state['latest_prediction']=pred; st.success(f'🎯 Predicted Customer Segment: **{pred}**')
                cc=cluster_col(df)
                if cc:
                    sd=df[df[cc]==pred]; a,b,c=st.columns(3); a.metric('Customers in Segment',len(sd)); b.metric('Average Income',f"{sd['Income'].mean():,.2f}" if 'Income' in sd else 'N/A'); c.metric('Average Spending',f"{sd['Total_spending'].mean():,.2f}" if 'Total_spending' in sd else 'N/A')
                st.subheader('📢 Marketing Recommendations')
                keys=[('marketing_message','Marketing Message'),('preferred_channel','Preferred Channel'),('personalized_offer','Personalized Offer'),('campaign_timing','Campaign Timing'),('discount_strategy','Discount Strategy'),('retention_strategy','Retention Strategy'),('primary_product','Primary Product'),('secondary_product','Secondary Product'),('cross_selling','Cross Selling'),('upselling','Upselling')]
                shown=False
                for k,t in keys:
                    if results.get(k) is not None: st.write(f'**{t}:** {results[k]}'); shown=True
                if not shown: st.info('No saved Day 3 recommendations found.')
            except Exception as e:
                st.error('❌ Prediction could not be completed.'); st.warning('The model may require the exact preprocessing pipeline used during training.'); st.code(str(e))

# VISUALIZATIONS
elif page=='📈 Visualizations':
    st.markdown('<div class="hero"><h1>📈 Customer Visualizations</h1><p>Explore customer behavior.</p></div>',unsafe_allow_html=True)
    if df is None: st.stop()
    nums=df.select_dtypes(include=np.number).columns.tolist()
    if nums:
        f=st.selectbox('Select Numeric Feature',nums); st.bar_chart(df[f].value_counts().sort_index().head(50)); st.dataframe(df[f].describe().to_frame(),use_container_width=True)

# MARKETING
elif page=='📢 Marketing Strategy':
    st.markdown('<div class="hero"><h1>📢 Marketing Strategy</h1><p>Business recommendations from Day 3.</p></div>',unsafe_allow_html=True)
    if not results: st.warning('day3_results.pkl not found. Run the Day 3 save cell first.')
    else:
        st.success('✅ Day 3 business insights loaded.')
        keys=[('marketing_message','Marketing Message'),('preferred_channel','Preferred Channel'),('personalized_offer','Personalized Offer'),('campaign_timing','Campaign Timing'),('discount_strategy','Discount Strategy'),('retention_strategy','Retention Strategy'),('primary_product','Primary Product'),('secondary_product','Secondary Product'),('cross_selling','Cross Selling'),('upselling','Upselling')]
        for k,t in keys:
            if results.get(k) is not None: st.markdown(f'### {t}'); st.info(str(results[k]))
        table=results.get('marketing_recommendation')
        if isinstance(table,pd.DataFrame): st.subheader('📋 Recommendation Table'); st.dataframe(table,use_container_width=True)

# DOWNLOAD
elif page=='⬇️ Download':
    st.markdown('<div class="hero"><h1>⬇️ Download</h1><p>Download project outputs.</p></div>',unsafe_allow_html=True)
    if df is not None: st.download_button('⬇️ Download Final Dataset',df.to_csv(index=False).encode(), 'Final_Dataset.csv','text/csv')
    if results_path and results:
        bio=io.BytesIO(); joblib.dump(results,bio); st.download_button('⬇️ Download Day 3 Results',bio.getvalue(),'day3_results.pkl','application/octet-stream')
    st.subheader('📁 Detected Files')
    for label, file_path in [
        ('Final Dataset.csv', csv_path),
        ('kmeans_model.pkl', model_path),
        ('day3_results.pkl', results_path)
    ]:
        if file_path:
            st.markdown(f"✅ **{label}**")
            st.text(str(file_path))
        else:
            st.markdown(f"❌ **{label}** not found")

st.sidebar.markdown('---'); st.sidebar.caption('Customer Personality Analysis | ML Project')