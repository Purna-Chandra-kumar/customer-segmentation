"""
generate_data.py
Generates a realistic synthetic customer dataset for segmentation.
Run: python generate_data.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 1000

# ── Demographic features ──────────────────────────────────────
ages      = np.concatenate([
    np.random.normal(28, 4, 200),   # young adults
    np.random.normal(38, 5, 350),   # mid adults
    np.random.normal(52, 6, 300),   # older adults
    np.random.normal(65, 5, 150),   # seniors
]).astype(int).clip(18, 80)
np.random.shuffle(ages)

incomes = np.concatenate([
    np.random.normal(28000,  5000, 250),
    np.random.normal(55000,  8000, 350),
    np.random.normal(90000, 12000, 250),
    np.random.normal(145000,20000, 150),
]).clip(15000, 200000).astype(int)
np.random.shuffle(incomes)

genders    = np.random.choice(['Male','Female','Other'], N, p=[0.48,0.49,0.03])
education  = np.random.choice(['High School','Bachelor','Master','PhD'], N, p=[0.30,0.40,0.22,0.08])
city_size  = np.random.choice(['Small','Medium','Large','Metro'], N, p=[0.15,0.25,0.35,0.25])
regions    = np.random.choice(['North','South','East','West','Central'], N)

# ── Behavioural features ─────────────────────────────────────
annual_spend   = (incomes * np.random.uniform(0.04, 0.18, N)).astype(int)
purchase_freq  = np.random.poisson(lam=(annual_spend / 3000).clip(1, 25))
avg_order_val  = (annual_spend / purchase_freq.clip(1)).astype(int)
recency_days   = np.random.exponential(scale=60, size=N).astype(int).clip(1, 365)
loyalty_years  = np.random.exponential(scale=3, size=N).round(1).clip(0.1, 15)
returns_pct    = np.random.beta(2, 8, N).round(3)          # 0–1
support_calls  = np.random.poisson(lam=1.5, size=N).clip(0,12)
email_open_rate= np.random.beta(3, 5, N).round(3)
mobile_user    = np.random.choice([0,1], N, p=[0.35, 0.65])

# ── Product preferences ───────────────────────────────────────
preferred_cat  = np.random.choice(
    ['Electronics','Clothing','Home & Garden','Sports','Beauty','Books','Food'],
    N, p=[0.20,0.18,0.15,0.12,0.14,0.11,0.10])
discount_seeker= np.random.choice([0,1], N, p=[0.55, 0.45])
subscription   = np.random.choice([0,1], N, p=[0.60, 0.40])

# ── Satisfaction ─────────────────────────────────────────────
satisfaction   = np.random.choice([1,2,3,4,5], N, p=[0.05,0.10,0.20,0.40,0.25])
nps_score      = np.random.randint(0, 11, N)

df = pd.DataFrame({
    'customer_id':       [f'C{str(i).zfill(5)}' for i in range(1, N+1)],
    'age':               ages,
    'gender':            genders,
    'annual_income':     incomes,
    'education':         education,
    'city_size':         city_size,
    'region':            regions,
    'annual_spend':      annual_spend,
    'purchase_frequency':purchase_freq,
    'avg_order_value':   avg_order_val,
    'recency_days':      recency_days,
    'loyalty_years':     loyalty_years,
    'returns_pct':       returns_pct,
    'support_calls':     support_calls,
    'email_open_rate':   email_open_rate,
    'mobile_user':       mobile_user,
    'preferred_category':preferred_cat,
    'discount_seeker':   discount_seeker,
    'has_subscription':  subscription,
    'satisfaction_score':satisfaction,
    'nps_score':         nps_score,
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/customers.csv', index=False)
print(f"✅  Generated {len(df)} customer records → data/customers.csv")
print(df.head())
