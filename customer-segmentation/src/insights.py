"""
insights.py
Generates a text-based business insights report from segmentation results.
"""
import pandas as pd
import numpy as np

df      = pd.read_csv('data/customers_segmented.csv')
profile = pd.read_csv('data/segment_profiles.csv', index_col=0)

print("=" * 65)
print("   CUSTOMER SEGMENTATION — BUSINESS INSIGHTS REPORT")
print("=" * 65)

total = len(df)
for i, row in profile.iterrows():
    pct = row['count'] / total * 100
    print(f"\n{'─'*65}")
    print(f"  SEGMENT {i+1}: {row['segment_name'].upper()}")
    print(f"  {int(row['count'])} customers ({pct:.1f}% of base)")
    print(f"{'─'*65}")
    print(f"  Demographics   : Avg age {row['avg_age']:.0f} | Income ${row['avg_income']:,.0f}/yr")
    print(f"  Spending       : ${row['avg_spend']:,.0f}/yr | {row['avg_freq']:.1f} orders/yr | ${row['avg_order_val']:,.0f} AOV")
    print(f"  Engagement     : {row['avg_recency']:.0f} days since last purchase | Loyalty {row['avg_loyalty']:.1f} yrs")
    print(f"  Satisfaction   : {row['avg_satisfaction']:.1f}/5 | NPS {row['avg_nps']:.0f}/10")
    print(f"  Digital        : {row['pct_mobile']*100:.0f}% mobile | {row['pct_subscription']*100:.0f}% subscribed")
    print(f"  Churn Risk     : {row['avg_churn_risk']*100:.0f}%")
    print(f"  Top Category   : {row['top_category']}")

    # Strategy recommendation
    print(f"\n  📌 Recommended Actions:")
    if 'Premium' in row['segment_name']:
        print("   • VIP loyalty programme with exclusive perks")
        print("   • Premium product bundles & early access")
        print("   • Dedicated account manager / concierge")
    elif 'Loyal' in row['segment_name']:
        print("   • Reward frequent purchases (points system)")
        print("   • Cross-sell complementary categories")
        print("   • Referral programme — high advocacy potential")
    elif 'Risk' in row['segment_name'] or 'Churn' in row['segment_name']:
        print("   • Immediate win-back campaign (discount + outreach)")
        print("   • Survey to understand dissatisfaction drivers")
        print("   • Triggered emails on inactivity (30/60/90 days)")
    else:
        print("   • Targeted promotions & limited-time discounts")
        print("   • Highlight value products in preferred category")
        print("   • Subscription upsell with first-month trial")

print(f"\n{'='*65}")
print("  KEY METRICS SUMMARY")
print(f"{'='*65}")
print(f"  Total customers       : {total:,}")
print(f"  Total annual revenue  : ${df['annual_spend'].sum():,.0f}")
print(f"  Avg spend per customer: ${df['annual_spend'].mean():,.0f}")
top_seg = profile.loc[profile['avg_spend'].idxmax(), 'segment_name']
risky   = profile.loc[profile['avg_churn_risk'].idxmax(), 'segment_name']
print(f"  Highest-value segment : {top_seg}")
print(f"  Highest churn risk    : {risky}")
print(f"{'='*65}\n")
