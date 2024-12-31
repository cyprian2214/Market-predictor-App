# Market Prediction Pro

A professional financial market prediction application built with Python and Streamlit. The app provides real-time analysis and predictions for multiple financial instruments including major indices, tech stocks, and forex pairs.

## Features

- **Multi-Market Analysis**: Monitor NASDAQ-100, S&P 500, Dow Jones, major tech stocks, and forex pairs
- **Technical Indicators**: 
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - EMA (Exponential Moving Average)
  - SMA (Simple Moving Average)
- **Real-time Predictions**: Machine learning-based price predictions updated every minute
- **Interactive Charts**: Professional-grade technical analysis charts with prediction indicators
- **Performance Tracking**: Historical prediction accuracy and signal strength metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Cyprian2214/Market-predictor-App.git
cd prediction_app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

3. Select markets to monitor from the sidebar
4. Click "Start" to begin monitoring and predictions
5. View real-time predictions and technical analysis

## Technical Details

- **Data Source**: Yahoo Finance API (yfinance)
- **Machine Learning**: Random Forest Regressor for price predictions
- **Frontend**: Streamlit for the web interface
- **Visualization**: Plotly for interactive charts

## Dependencies

- Python 3.12+
- streamlit
- pandas
- numpy
- yfinance
- scikit-learn
- plotly
- ta (Technical Analysis library)

## License

MIT License
