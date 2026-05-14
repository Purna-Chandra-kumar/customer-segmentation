"""
visualize.py
Generates all segmentation visualizations:
  - PCA scatter plot (clusters)
  - Elbow + Silhouette curves
  - Radar/spider chart per segment
  - Feature importance bar chart
  - Segment size donut chart
  - Heatmap of segment profiles
Run AFTER segmentation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────
BG   = '#0d0d14'
BG1  = '#13131e'
BG2  = '#1a1a28'
GRID = '#2a2a3a'
T1   = '#eeeef8'
T2   = '#9090b0'
COLORS = ['#b8e030','#4facfe','#fb923c','#a78bfa']

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG1,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   T2,
    'xtick.color':       T2,
    'ytick.color':       T2,
    'text.color':        T1,
    'grid.color':        GRID,
    'grid.linewidth':    0.5,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
})

os.makedirs('outputs', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
df      = pd.read_csv('data/customers_segmented.csv')
profile = pd.read_csv('data/segment_profiles.csv', index_col=0)
elbow   = pd.read_csv('data/elbow_data.csv')

seg_names  = profile['segment_name'].tolist()
seg_counts = profile['count'].tolist()

# ═══════════════════════════════════════════════════════════════
# 1. MASTER DASHBOARD (4-panel)
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 13), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32,
                         left=0.06, right=0.97, top=0.91, bottom=0.06)

ax_scatter = fig.add_subplot(gs[0, :2])
ax_donut   = fig.add_subplot(gs[0, 2])
ax_elbow   = fig.add_subplot(gs[1, 0])
ax_heat    = fig.add_subplot(gs[1, 1:])

fig.suptitle('Customer Segmentation Dashboard', fontsize=18,
             fontweight='bold', color=T1, y=0.96)

# ── Panel 1: PCA Scatter ──────────────────────────────────────
for i, (name, color) in enumerate(zip(seg_names, COLORS)):
    mask = df['cluster'] == i
    ax_scatter.scatter(df.loc[mask,'pca_x'], df.loc[mask,'pca_y'],
                       c=color, s=18, alpha=0.65, linewidths=0, label=name)

ax_scatter.set_title('Customer Clusters (PCA 2D Projection)', color=T1, pad=10)
ax_scatter.set_xlabel('PC1'); ax_scatter.set_ylabel('PC2')
ax_scatter.grid(True, alpha=0.3)
ax_scatter.legend(loc='upper right', fontsize=8,
                  facecolor=BG2, edgecolor=GRID, labelcolor=T1)

# ── Panel 2: Donut ────────────────────────────────────────────
wedges, texts, autotexts = ax_donut.pie(
    seg_counts, colors=COLORS, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    autopct='%1.1f%%', pctdistance=0.78
)
for at in autotexts:
    at.set(color=BG, fontsize=9, fontweight='bold')
ax_donut.set_title('Segment Distribution', color=T1, pad=10)
legend_patches = [mpatches.Patch(color=c, label=f'{n}\n({cnt})')
                  for c,n,cnt in zip(COLORS, seg_names, seg_counts)]
ax_donut.legend(handles=legend_patches, loc='lower center',
                bbox_to_anchor=(0.5,-0.22), fontsize=7.5,
                facecolor=BG2, edgecolor=GRID, labelcolor=T1, ncol=2)

# ── Panel 3: Elbow ────────────────────────────────────────────
ax2 = ax_elbow.twinx()
ax_elbow.plot(elbow['k'], elbow['inertia']/1000, 'o-',
              color='#4facfe', linewidth=2, markersize=5, label='Inertia (K)')
ax2.plot(elbow['k'], elbow['silhouette'], 's--',
         color='#b8e030', linewidth=2, markersize=5, label='Silhouette')
ax_elbow.set_title('Optimal k — Elbow & Silhouette', color=T1, pad=10)
ax_elbow.set_xlabel('Number of Clusters k')
ax_elbow.set_ylabel('Inertia (×1000)', color='#4facfe')
ax2.set_ylabel('Silhouette Score', color='#b8e030')
ax2.tick_params(colors='#b8e030')
ax_elbow.axvline(4, color='#fb923c', linestyle=':', alpha=0.8, linewidth=1.5)
ax_elbow.grid(True, alpha=0.3)
lines1,l1 = ax_elbow.get_legend_handles_labels()
lines2,l2 = ax2.get_legend_handles_labels()
ax_elbow.legend(lines1+lines2, l1+l2, fontsize=8,
                facecolor=BG2, edgecolor=GRID, labelcolor=T1)

# ── Panel 4: Heatmap ──────────────────────────────────────────
heat_cols = ['avg_income','avg_spend','avg_freq','avg_recency',
             'avg_loyalty','avg_satisfaction','avg_churn_risk','pct_mobile']
heat_labels = ['Income','Spend','Frequency','Recency',
               'Loyalty Yrs','Satisfaction','Churn Risk','Mobile %']

heat_data = profile[heat_cols].values.astype(float)
heat_norm  = (heat_data - heat_data.min(axis=0)) / (np.ptp(heat_data, axis=0) + 1e-9)

im = ax_heat.imshow(heat_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax_heat.set_xticks(range(len(heat_labels)))
ax_heat.set_xticklabels(heat_labels, rotation=30, ha='right', fontsize=9)
ax_heat.set_yticks(range(len(seg_names)))
ax_heat.set_yticklabels([f'{n}' for n in seg_names], fontsize=9)
ax_heat.set_title('Segment Profile Heatmap (normalised)', color=T1, pad=10)

for i in range(len(seg_names)):
    for j in range(len(heat_labels)):
        val = heat_data[i, j]
        txt = f'{val:.0f}' if val > 1 else f'{val:.2f}'
        ax_heat.text(j, i, txt, ha='center', va='center',
                     fontsize=7.5, color='#111' if heat_norm[i,j] > 0.55 else T1)

plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color=T2)

plt.savefig('outputs/dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅  outputs/dashboard.png")

# ═══════════════════════════════════════════════════════════════
# 2. RADAR CHART — segment DNA
# ═══════════════════════════════════════════════════════════════
radar_cols  = ['avg_spend','avg_freq','avg_loyalty','avg_satisfaction','pct_mobile','pct_subscription','avg_nps']
radar_labs  = ['Spend','Frequency','Loyalty','Satisfaction','Mobile','Subscription','NPS']
N_r = len(radar_labs)
angles = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                          subplot_kw=dict(polar=True), facecolor=BG)
fig.suptitle('Segment DNA — Radar Profiles', fontsize=15, fontweight='bold', color=T1, y=1.02)

radar_data = profile[radar_cols].values.astype(float)
radar_norm = (radar_data - radar_data.min(axis=0)) / (np.ptp(radar_data, axis=0) + 1e-9)

for idx, ax in enumerate(axes):
    vals = radar_norm[idx].tolist() + [radar_norm[idx][0]]
    ax.set_facecolor(BG1)
    ax.plot(angles, vals, color=COLORS[idx], linewidth=2)
    ax.fill(angles, vals, color=COLORS[idx], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labs, size=9, color=T2)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines['polar'].set_color(GRID)
    ax.set_title(seg_names[idx], color=COLORS[idx], pad=14, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/radar_profiles.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅  outputs/radar_profiles.png")

# ═══════════════════════════════════════════════════════════════
# 3. FEATURE IMPORTANCE (variance across segments)
# ═══════════════════════════════════════════════════════════════
feat_cols = ['avg_income','avg_spend','avg_freq','avg_order_val',
             'avg_recency','avg_loyalty','avg_satisfaction','avg_churn_risk',
             'pct_mobile','pct_subscription','pct_discount','avg_nps']
feat_labels = ['Income','Spend','Frequency','Avg Order',
               'Recency','Loyalty','Satisfaction','Churn Risk',
               'Mobile','Subscription','Discount','NPS']

feat_data = profile[feat_cols].values.astype(float)
feat_norm = (feat_data - feat_data.min(axis=0)) / (np.ptp(feat_data, axis=0) + 1e-9)
importance = feat_norm.std(axis=0)
order = np.argsort(importance)[::-1]

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
bars = ax.barh([feat_labels[i] for i in order],
               [importance[i] for i in order],
               color=[COLORS[i % 4] for i in range(len(order))],
               edgecolor='none', height=0.6)
ax.set_title('Feature Discriminability Across Segments', color=T1, fontsize=13, pad=10)
ax.set_xlabel('Std Dev of Normalised Value (higher = more discriminating)')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
for bar, val in zip(bars, [importance[i] for i in order]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', color=T1, fontsize=9)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅  outputs/feature_importance.png")

# ═══════════════════════════════════════════════════════════════
# 4. SEGMENT INCOME vs SPEND scatter
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

# Income vs Spend
for i, (name, color) in enumerate(zip(seg_names, COLORS)):
    mask = df['cluster'] == i
    axes[0].scatter(df.loc[mask,'annual_income']/1000, df.loc[mask,'annual_spend']/1000,
                    c=color, s=14, alpha=0.55, label=name, linewidths=0)
axes[0].set_title('Income vs Annual Spend by Segment', color=T1, pad=10)
axes[0].set_xlabel('Annual Income ($K)'); axes[0].set_ylabel('Annual Spend ($K)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=T1)

# Recency vs Frequency
for i, (name, color) in enumerate(zip(seg_names, COLORS)):
    mask = df['cluster'] == i
    axes[1].scatter(df.loc[mask,'recency_days'], df.loc[mask,'purchase_frequency'],
                    c=color, s=14, alpha=0.55, label=name, linewidths=0)
axes[1].set_title('Recency vs Purchase Frequency', color=T1, pad=10)
axes[1].set_xlabel('Days Since Last Purchase'); axes[1].set_ylabel('Purchase Frequency / yr')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=T1)

plt.tight_layout(pad=2)
plt.savefig('outputs/scatter_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅  outputs/scatter_analysis.png")

print("\n🎉 All visualizations saved to outputs/")
