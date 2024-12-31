@echo off
echo Creating new virtual environment...
python -m venv venv --clear

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install streamlit pandas numpy yfinance scikit-learn plotly ta requests beautifulsoup4 schedule

echo Verifying streamlit installation...
where streamlit

echo Setup complete! Run start_app.bat to start the application.
pause
