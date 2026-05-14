"""
app.py  —  Interactive Streamlit Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0d0d14; }
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: #1a1a28; border: 1px solid #2e2e48;
    border-radius: 10px; padding: 16px 20px; text-align: center;
  }
  .seg-card {
    background: #13131e; border-left: 4px solid;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
  }
  h1,h2,h3 { color: #eeeef8 !important; }
</style>""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load():
    df      = pd.read_csv('data/customers_segmented.csv')
    profile = pd.read_csv('data/segment_profiles.csv', index_col=0)
    elbow   = pd.read_csv('data/elbow_data.csv')
    return df, profile, elbow

df, profile, elbow = load()
COLORS = ['#b8e030','#4facfe','#fb923c','#a78bfa']
seg_names = profile['segment_name'].tolist()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 🎯 Customer Segmentation")
st.sidebar.markdown("---")

selected_segs = st.sidebar.multiselect(
    "Filter Segments", seg_names, default=seg_names
)
sel_indices = [i for i, n in enumerate(seg_names) if n in selected_segs]
df_f = df[df['cluster'].isin(sel_indices)].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
age_range = st.sidebar.slider("Age Range", 18, 80, (18, 80))
inc_range = st.sidebar.slider("Income Range ($K)", 15, 200, (15, 200))
df_f = df_f[
    df_f['age'].between(*age_range) &
    (df_f['annual_income'] / 1000).between(*inc_range)
]
st.sidebar.markdown(f"**{len(df_f):,}** customers selected")

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🎯 Customer Segmentation Dashboard")
st.markdown(f"Analysing **{len(df_f):,}** customers across **{len(selected_segs)}** segments")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers",   f"{len(df_f):,}")
k2.metric("Avg Annual Spend",  f"${df_f['annual_spend'].mean():,.0f}")
k3.metric("Avg Order Value",   f"${df_f['avg_order_value'].mean():,.0f}")
k4.metric("Avg Satisfaction",  f"{df_f['satisfaction_score'].mean():.1f} / 5")
k5.metric("Avg Churn Risk",    f"{df_f['churn_risk'].mean()*100:.0f}%")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Deep Dive", "📈 Trends", "📋 Data"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Customer Clusters (PCA Projection)")
        fig_pca = px.scatter(
            df_f, x='pca_x', y='pca_y',
            color=df_f['cluster'].map(dict(enumerate(seg_names))),
            color_discrete_sequence=COLORS,
            opacity=0.65, size_max=6,
            labels={'color':'Segment','pca_x':'PC1','pca_y':'PC2'},
            template='plotly_dark'
        )
        fig_pca.update_traces(marker=dict(size=5))
        fig_pca.update_layout(
            paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
            height=380, legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_r:
        st.subheader("Segment Distribution")
        fig_pie = px.pie(
            values=profile.loc[sel_indices,'count'],
            names=[seg_names[i] for i in sel_indices],
            color_discrete_sequence=[COLORS[i] for i in sel_indices],
            hole=0.55, template='plotly_dark'
        )
        fig_pie.update_layout(
            paper_bgcolor='#13131e', height=380,
            legend=dict(orientation='h', y=-0.2, font=dict(size=10))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Segment Profile Heatmap")
    heat_cols = ['avg_income','avg_spend','avg_freq','avg_recency',
                 'avg_loyalty','avg_satisfaction','avg_churn_risk','pct_mobile','pct_subscription']
    heat_labs  = ['Income','Spend','Frequency','Recency','Loyalty','Satisfaction','Churn Risk','Mobile','Subscription']
    heat_data  = profile.loc[sel_indices, heat_cols].values.astype(float)
    heat_norm  = (heat_data - heat_data.min(0)) / (heat_data.ptp(0) + 1e-9)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_norm,
        x=heat_labs,
        y=[seg_names[i] for i in sel_indices],
        colorscale='RdYlGn', showscale=True,
        text=[[f'{v:.1f}' for v in row] for row in heat_data],
        texttemplate='%{text}', textfont=dict(size=10)
    ))
    fig_heat.update_layout(
        paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
        height=220, margin=dict(l=160, r=20, t=20, b=60)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Income vs Annual Spend")
        fig_sc1 = px.scatter(
            df_f, x='annual_income', y='annual_spend',
            color=df_f['cluster'].map(dict(enumerate(seg_names))),
            color_discrete_sequence=COLORS, opacity=0.55,
            labels={'color':'Segment','annual_income':'Annual Income ($)','annual_spend':'Annual Spend ($)'},
            template='plotly_dark'
        )
        fig_sc1.update_traces(marker=dict(size=5))
        fig_sc1.update_layout(paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
                              height=340, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_b:
        st.subheader("Recency vs Purchase Frequency")
        fig_sc2 = px.scatter(
            df_f, x='recency_days', y='purchase_frequency',
            color=df_f['cluster'].map(dict(enumerate(seg_names))),
            color_discrete_sequence=COLORS, opacity=0.55,
            labels={'color':'Segment','recency_days':'Days Since Last Purchase','purchase_frequency':'Purchases / Year'},
            template='plotly_dark'
        )
        fig_sc2.update_traces(marker=dict(size=5))
        fig_sc2.update_layout(paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
                              height=340, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.subheader("Radar — Segment DNA")
    radar_cols = ['avg_spend','avg_freq','avg_loyalty','avg_satisfaction',
                  'pct_mobile','pct_subscription','avg_nps']
    radar_labs = ['Spend','Frequency','Loyalty','Satisfaction','Mobile','Subscription','NPS']
    radar_data = profile.loc[sel_indices, radar_cols].values.astype(float)
    radar_norm = (radar_data - radar_data.min(0)) / (radar_data.ptp(0) + 1e-9)

    fig_rad = go.Figure()
    for idx, i in enumerate(sel_indices):
        vals = radar_norm[idx].tolist() + [radar_norm[idx][0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labs + [radar_labs[0]],
            fill='toself', name=seg_names[i],
            line=dict(color=COLORS[i], width=2),
            fillcolor=COLORS[i].replace('#', 'rgba(').rstrip(')') if '#' in COLORS[i] else COLORS[i],
            opacity=0.3
        ))
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], color='#5a5a78'),
                   bgcolor='#1a1a28', angularaxis=dict(color='#9090b0')),
        paper_bgcolor='#13131e', showlegend=True, height=380,
        legend=dict(orientation='h', y=-0.15)
    )
    st.plotly_chart(fig_rad, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — TRENDS
# ══════════════════════════════════════════════════════════════
with tab3:
    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Elbow & Silhouette Curve")
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(go.Scatter(x=elbow['k'], y=elbow['inertia']/1000,
                                       mode='lines+markers', name='Inertia (K)',
                                       line=dict(color='#4facfe', width=2)), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(x=elbow['k'], y=elbow['silhouette'],
                                       mode='lines+markers', name='Silhouette',
                                       line=dict(color='#b8e030', width=2, dash='dash')), secondary_y=True)
        fig_elbow.add_vline(x=4, line_dash='dot', line_color='#fb923c', annotation_text='k=4')
        fig_elbow.update_layout(paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
                                 height=320, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_f:
        st.subheader("Preferred Categories by Segment")
        cat_data = df_f.groupby(['cluster','preferred_category']).size().reset_index(name='count')
        cat_data['segment'] = cat_data['cluster'].map(dict(enumerate(seg_names)))
        fig_cat = px.bar(cat_data, x='preferred_category', y='count', color='segment',
                         barmode='group', color_discrete_sequence=COLORS, template='plotly_dark')
        fig_cat.update_layout(paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
                               height=320, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Age Distribution by Segment")
    fig_age = px.histogram(df_f, x='age', color=df_f['cluster'].map(dict(enumerate(seg_names))),
                            nbins=30, barmode='overlay', opacity=0.65,
                            color_discrete_sequence=COLORS, template='plotly_dark',
                            labels={'color':'Segment'})
    fig_age.update_layout(paper_bgcolor='#13131e', plot_bgcolor='#1a1a28',
                           height=300, legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig_age, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — DATA TABLE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Segmented Customer Data")
    show_cols = ['customer_id','age','annual_income','annual_spend',
                 'purchase_frequency','recency_days','satisfaction_score',
                 'preferred_category','cluster']
    display = df_f[show_cols].copy()
    display['segment'] = display['cluster'].map(dict(enumerate(seg_names)))
    display = display.drop('cluster', axis=1)
    st.dataframe(display, use_container_width=True, height=420)

    csv = display.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "segmented_customers.csv", "text/csv")
