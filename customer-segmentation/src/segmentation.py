"""
segmentation.py
Full customer segmentation pipeline:
  1. Load & preprocess data
  2. Feature engineering
  3. Optimal k selection (Elbow + Silhouette)
  4. KMeans clustering
  5. Segment profiling & naming
  6. Export results
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings, os
warnings.filterwarnings('ignore')

# ── 1. Load data ──────────────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv('data/customers.csv')
print(f"   {len(df)} rows, {df.shape[1]} columns")

# ── 2. Feature engineering ────────────────────────────────────
print("⚙️  Engineering features...")

df['spend_to_income_ratio'] = df['annual_spend'] / df['annual_income']
df['value_score']           = (df['avg_order_value'] * df['purchase_frequency']) / 1000
df['engagement_score']      = df['email_open_rate'] * (1 - df['recency_days'] / 365)
df['churn_risk']            = (
    (df['recency_days'] / 365) * 0.4 +
    (1 - df['satisfaction_score'] / 5) * 0.4 +
    (df['support_calls'] / 12) * 0.2
)

# ── 3. Prepare feature matrix ─────────────────────────────────
NUM_FEATURES = [
    'age', 'annual_income', 'annual_spend', 'purchase_frequency',
    'avg_order_value', 'recency_days', 'loyalty_years', 'returns_pct',
    'support_calls', 'email_open_rate', 'satisfaction_score', 'nps_score',
    'spend_to_income_ratio', 'value_score', 'engagement_score', 'churn_risk'
]

X = df[NUM_FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for 2D visualisation
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]
print(f"   PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── 4. Find optimal k ─────────────────────────────────────────
print("🔍 Finding optimal number of clusters...")
inertias, sil_scores = [], []
K_RANGE = range(2, 10)

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=500))

best_k_sil = K_RANGE[np.argmax(sil_scores)]
print(f"   Best k by silhouette: {best_k_sil} (score={max(sil_scores):.3f})")

# Use 4 segments — balances business interpretability & silhouette
K = 4
print(f"   Using K={K} segments")

# ── 5. Final clustering ───────────────────────────────────────
print("🎯 Running KMeans clustering...")
km_final = KMeans(n_clusters=K, init='k-means++', n_init=20, random_state=42)
df['cluster'] = km_final.fit_predict(X_scaled)

# ── 6. Profile each segment ───────────────────────────────────
print("📊 Profiling segments...")
profile = df.groupby('cluster').agg(
    count            = ('customer_id','count'),
    avg_age          = ('age','mean'),
    avg_income       = ('annual_income','mean'),
    avg_spend        = ('annual_spend','mean'),
    avg_freq         = ('purchase_frequency','mean'),
    avg_order_val    = ('avg_order_value','mean'),
    avg_recency      = ('recency_days','mean'),
    avg_loyalty      = ('loyalty_years','mean'),
    avg_satisfaction = ('satisfaction_score','mean'),
    avg_nps          = ('nps_score','mean'),
    avg_churn_risk   = ('churn_risk','mean'),
    pct_mobile       = ('mobile_user','mean'),
    pct_subscription = ('has_subscription','mean'),
    pct_discount     = ('discount_seeker','mean'),
    top_category     = ('preferred_category', lambda x: x.mode()[0])
).round(2)

# ── 7. Auto-name segments ─────────────────────────────────────
def name_segment(row):
    if row['avg_spend'] > 8000 and row['avg_income'] > 90000:
        return 'Premium High-Value'
    elif row['avg_freq'] > 10 and row['avg_recency'] < 40:
        return 'Loyal Frequent Buyers'
    elif row['avg_churn_risk'] > 0.45:
        return 'At-Risk / Churners'
    else:
        return 'Occasional Bargain Hunters'

profile['segment_name'] = profile.apply(name_segment, axis=1)

SEGMENT_COLORS = {0:'#b8e030', 1:'#4facfe', 2:'#fb923c', 3:'#a78bfa'}
profile['color'] = [SEGMENT_COLORS[i] for i in profile.index]

print("\n── Segment Summary ───────────────────────────────────────")
print(profile[['segment_name','count','avg_income','avg_spend','avg_freq','avg_churn_risk']].to_string())

# ── 8. Elbow data ─────────────────────────────────────────────
elbow_df = pd.DataFrame({
    'k': list(K_RANGE),
    'inertia': inertias,
    'silhouette': sil_scores
})

# ── 9. Save outputs ───────────────────────────────────────────
os.makedirs('data', exist_ok=True)
df.to_csv('data/customers_segmented.csv', index=False)
profile.to_csv('data/segment_profiles.csv')
elbow_df.to_csv('data/elbow_data.csv', index=False)
print("\n✅ Outputs saved:")
print("   data/customers_segmented.csv")
print("   data/segment_profiles.csv")
print("   data/elbow_data.csv")
