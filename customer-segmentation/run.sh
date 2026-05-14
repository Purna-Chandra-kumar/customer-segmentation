#!/bin/bash
echo "============================================"
echo "  Customer Segmentation Project"
echo "============================================"

echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[2/4] Generating customer data..."
python src/generate_data.py

echo ""
echo "[3/4] Running segmentation pipeline..."
python src/segmentation.py

echo ""
echo "[4/4] Generating visualizations..."
python src/visualize.py

echo ""
echo "============================================"
echo "  Charts saved to outputs/"
echo "  Starting Streamlit dashboard..."
echo "============================================"
streamlit run app.py
