@echo off
echo Starting debug process...
echo Current directory:
cd
echo.
echo Activating virtual environment...
call venv\Scripts\activate
echo.
echo Python version:
python --version
echo.
echo Starting Streamlit...
streamlit run debug_app.py --server.port 8505 --server.headless true
pause
