@echo off
echo Starting Market Prediction App...
call venv\Scripts\activate
venv\Scripts\streamlit run app.py --server.port 8505 --server.headless true
pause
