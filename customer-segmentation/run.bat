@echo off
echo ============================================
echo   Customer Segmentation Project
echo ============================================

echo.
echo [1/4] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [2/4] Generating customer data...
cd /d "%~dp0"
python src/generate_data.py

echo.
echo [3/4] Running segmentation pipeline...
python src/segmentation.py

echo.
echo [4/4] Generating visualizations...
python src/visualize.py

echo.
echo ============================================
echo   Done! Charts saved to outputs/
echo   Starting Streamlit dashboard...
echo ============================================
echo.
streamlit run app.py
pause
