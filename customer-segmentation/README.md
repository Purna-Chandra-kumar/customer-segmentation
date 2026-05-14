# 🎯 Customer Segmentation Project

An end-to-end customer analytics pipeline using **KMeans clustering** to segment 1,000 customers into actionable groups based on demographics, behaviour, and preferences.

---

## 📁 Project Structure

```
customer-segmentation/
├── app.py                        ← Streamlit interactive dashboard
├── requirements.txt
├── run.bat                       ← Windows one-click launcher
├── run.sh                        ← Mac/Linux one-click launcher
├── src/
│   ├── generate_data.py          ← Synthetic dataset (1,000 customers)
│   ├── segmentation.py           ← KMeans pipeline + segment profiling
│   ├── visualize.py              ← Static charts (PNG output)
│   └── insights.py               ← Business insights text report
├── data/                         ← Auto-generated CSV files
│   ├── customers.csv
│   ├── customers_segmented.csv
│   ├── segment_profiles.csv
│   └── elbow_data.csv
└── outputs/                      ← Auto-generated chart PNGs
    ├── dashboard.png
    ├── radar_profiles.png
    ├── feature_importance.png
    └── scatter_analysis.png
```

---

## 🚀 Quick Start (Windows)

```cmd
# Just double-click run.bat  OR:
pip install -r requirements.txt
python src/generate_data.py
python src/segmentation.py
python src/visualize.py
streamlit run app.py
```

## 🚀 Quick Start (Mac/Linux)

```bash
chmod +x run.sh && ./run.sh
```

---

## 🔬 Methodology

| Step | Detail |
|------|--------|
| **Data** | 1,000 synthetic customers with 21 features |
| **Features** | Demographics + behavioural + product preferences |
| **Preprocessing** | StandardScaler normalisation |
| **Optimal k** | Elbow method + Silhouette score → k=4 |
| **Algorithm** | KMeans++ (20 initialisations, random_state=42) |
| **Visualisation** | PCA 2D projection, Radar charts, Heatmap |

---

## 👥 The 4 Customer Segments

| # | Segment | Key Trait | Strategy |
|---|---------|-----------|----------|
| 1 | **Premium High-Value** | High income + high spend | VIP programme, early access |
| 2 | **Loyal Frequent Buyers** | High frequency + low recency | Rewards, referral programme |
| 3 | **At-Risk / Churners** | High churn risk + low satisfaction | Win-back campaign, surveys |
| 4 | **Occasional Bargain Hunters** | Low frequency + discount-seeking | Targeted promotions, upsell |

---

## 📊 Features Used for Clustering

- Age, Annual Income, Education, City Size
- Annual Spend, Purchase Frequency, Avg Order Value
- Recency (days since last purchase), Loyalty Years
- Email Open Rate, Mobile Usage, Subscription Status
- Satisfaction Score, NPS, Support Calls, Returns %
- Engineered: Spend/Income Ratio, Value Score, Churn Risk

---

## 🛠 Tech Stack

- **Python 3.10+**
- **scikit-learn** — KMeans, PCA, StandardScaler, Silhouette
- **pandas / numpy** — data wrangling
- **matplotlib** — static chart exports
- **plotly** — interactive charts
- **streamlit** — interactive web dashboard

---

## 📈 Outputs

After running, you get:
- `outputs/dashboard.png` — 4-panel master dashboard
- `outputs/radar_profiles.png` — Segment DNA radar charts
- `outputs/feature_importance.png` — Discriminating features
- `outputs/scatter_analysis.png` — Income/Spend & Recency/Frequency
- `data/customers_segmented.csv` — Full dataset with cluster labels
- `data/segment_profiles.csv` — Aggregated segment statistics
- Live Streamlit app at http://localhost:8501

---

## 📄 License

MIT
