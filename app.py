import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
import threading

# Set page configuration
st.set_page_config(
    page_title="Market Prediction Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0083B8;
    }
    .stButton>button:hover {
        background-color: #00405d;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    .prediction-up {
        color: #00ff00;
        font-weight: bold;
    }
    .prediction-down {
        color: #ff0000;
        font-weight: bold;
    }
    .market-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

class MarketPredictionApp:
    def __init__(self):
        self.symbols = {
            'NASDAQ-100': '^NDX',
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Google': 'GOOGL',
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'JPY=X'
        }
        self.models = {}
        self.is_running = False
        self.predictions = {}
        
    def fetch_data(self, symbol, period='1d', interval='1m'):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def prepare_features(self, df):
        if df is None or len(df) < 50:
            return None, None
            
        # Technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['BB_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_middle'] = ta.volatility.BollingerBands(df['Close']).bollinger_mavg()
        df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        
        # Create features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 
                   'BB_upper', 'BB_middle', 'BB_lower', 'EMA_20', 'SMA_50']
        df = df.dropna()
        
        if len(df) < 2:
            return None, None
            
        X = df[features].values[:-1]
        y = df['Close'].values[1:]
        
        return X, y, df

    def create_technical_chart(self, df, symbol_name, prediction=None, current_price=None):
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])

        # Add candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='Price'),
                     row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'],
                                line=dict(color='gray', width=1),
                                name='BB Upper', opacity=0.3),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'],
                                line=dict(color='gray', width=1),
                                name='BB Lower', opacity=0.3,
                                fill='tonexty'),
                     row=1, col=1)

        # Add EMA and SMA
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'],
                                line=dict(color='blue', width=1),
                                name='EMA 20'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                                line=dict(color='orange', width=1),
                                name='SMA 50'),
                     row=1, col=1)

        # Add MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                                line=dict(color='blue', width=1),
                                name='MACD'),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'],
                                line=dict(color='orange', width=1),
                                name='Signal'),
                     row=2, col=1)

        # Add prediction arrow if available
        if prediction is not None and current_price is not None:
            last_index = df.index[-1]
            next_index = last_index + pd.Timedelta(minutes=1)
            
            arrow_color = 'green' if prediction > current_price else 'red'
            arrow_symbol = '⬆' if prediction > current_price else '⬇'
            
            # Add arrow annotation
            fig.add_annotation(
                x=last_index,
                y=current_price,
                text=arrow_symbol,
                showarrow=True,
                arrowhead=1,
                arrowsize=2,
                arrowwidth=3,
                arrowcolor=arrow_color,
                ax=0,
                ay=-40 if prediction > current_price else 40
            )
            
            # Add predicted price line
            fig.add_trace(go.Scatter(
                x=[last_index, next_index],
                y=[current_price, prediction],
                mode='lines',
                line=dict(color=arrow_color, dash='dash'),
                name='Prediction'
            ), row=1, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol_name} Technical Analysis',
            yaxis_title='Price',
            yaxis2_title='MACD',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    def train_model(self, symbol):
        df = self.fetch_data(symbol, period='5d', interval='1m')
        X, y, _ = self.prepare_features(df)
        
        if X is None or y is None:
            return None
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def make_prediction(self, symbol):
        try:
            # Get latest data
            df = self.fetch_data(symbol, period='1d', interval='1m')
            if df is None or len(df) < 50:
                return None, None, None
                
            X, _, df_with_features = self.prepare_features(df)
            if X is None or len(X) == 0:
                return None, None, None
                
            # Get or train model
            if symbol not in self.models:
                self.models[symbol] = self.train_model(symbol)
                
            if self.models[symbol] is None:
                return None, None, None
                
            # Make prediction
            latest_features = X[-1].reshape(1, -1)
            prediction = self.models[symbol].predict(latest_features)[0]
            current_price = df['Close'].iloc[-1]
            
            return prediction, current_price, df_with_features
            
        except Exception as e:
            st.error(f"Error making prediction for {symbol}: {str(e)}")
            return None, None, None

    def continuous_prediction(self, selected_symbols):
        while self.is_running:
            for symbol_name in selected_symbols:
                symbol = self.symbols[symbol_name]
                prediction, current_price, df = self.make_prediction(symbol)
                
                if prediction is not None and current_price is not None:
                    if symbol not in self.predictions:
                        self.predictions[symbol] = []
                        
                    self.predictions[symbol].append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': symbol_name,
                        'current_price': current_price,
                        'predicted_price': prediction,
                        'predicted_movement': 'Up' if prediction > current_price else 'Down',
                        'df': df
                    })
                    
                    # Keep only last 100 predictions
                    if len(self.predictions[symbol]) > 100:
                        self.predictions[symbol].pop(0)
                        
            time.sleep(60)  # Update every minute

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Market Prediction Pro</h1>
            <p style='font-size: 1.2em; color: #666;'>Advanced Technical Analysis & Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    app = MarketPredictionApp()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Settings")
        st.markdown("---")
        
        # Market selection with categories
        st.markdown("#### Select Markets")
        
        # Indices
        st.markdown("##### Major Indices")
        indices = st.multiselect(
            "Select Major Indices",
            ['NASDAQ-100', 'S&P 500', 'Dow Jones'],
            default=['NASDAQ-100'],
            label_visibility="collapsed"
        )
        
        # Stocks
        st.markdown("##### Tech Stocks")
        stocks = st.multiselect(
            "Select Tech Stocks",
            ['Apple', 'Microsoft', 'Google'],
            default=['Apple'],
            label_visibility="collapsed"
        )
        
        # Forex
        st.markdown("##### Forex Pairs")
        forex = st.multiselect(
            "Select Forex Pairs",
            ['EUR/USD', 'GBP/USD', 'USD/JPY'],
            default=[],
            label_visibility="collapsed"
        )
        
        selected_symbols = indices + stocks + forex
        
        st.markdown("---")
        st.markdown("#### Update Interval")
        update_interval = st.slider("Minutes", 1, 5, 1)
        
        # Control buttons with better styling
        st.markdown("#### Controls")
        start_col, stop_col = st.columns(2)
        with start_col:
            start_button = st.button("▶ Start")
        with stop_col:
            stop_button = st.button("⏹ Stop")
    
    # Main content
    if not selected_symbols:
        st.warning("🎯 Please select at least one market to monitor")
        return
        
    # Initialize or update app state
    if start_button:
        app.is_running = True
        app.predictions = {}
        prediction_thread = threading.Thread(
            target=app.continuous_prediction,
            args=(selected_symbols,)
        )
        prediction_thread.start()
        st.success("✅ Started monitoring selected markets")
        
    if stop_button:
        app.is_running = False
        st.info("🛑 Stopped monitoring markets")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Live Analysis", "📊 Historical Data"])
    
    with tab1:
        # Display predictions in a grid
        for symbol_name in selected_symbols:
            symbol = app.symbols[symbol_name]
            if symbol in app.predictions and app.predictions[symbol]:
                with st.container():
                    st.markdown(f"""
                        <div class='market-header'>
                            <h3>{symbol_name}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Get latest prediction
                    latest = app.predictions[symbol][-1]
                    
                    # Display metrics with improved styling
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${latest['current_price']:.2f}")
                    
                    with col2:
                        st.metric("Predicted Price", f"${latest['predicted_price']:.2f}")
                    
                    with col3:
                        movement_color = "prediction-up" if latest['predicted_movement'] == "Up" else "prediction-down"
                        st.markdown(f"""
                            <div style='text-align: center;'>
                                <p>Predicted Movement</p>
                                <p class='{movement_color}'>{latest['predicted_movement']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        confidence = abs(latest['predicted_price'] - latest['current_price']) / latest['current_price'] * 100
                        st.metric("Signal Strength", f"{confidence:.1f}%")
                    
                    # Create and display technical chart
                    fig = app.create_technical_chart(
                        latest['df'],
                        symbol_name,
                        latest['predicted_price'],
                        latest['current_price']
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        for symbol_name in selected_symbols:
            symbol = app.symbols[symbol_name]
            if symbol in app.predictions and app.predictions[symbol]:
                st.subheader(f"{symbol_name} - Historical Predictions")
                history_df = pd.DataFrame([p for p in app.predictions[symbol] if 'df' not in p])
                
                # Calculate accuracy metrics
                history_df['actual_movement'] = history_df['current_price'].diff().gt(0).shift(-1)
                history_df['prediction_correct'] = (
                    history_df['predicted_movement'] == 'Up'
                ) == history_df['actual_movement']
                
                # Display accuracy metrics
                accuracy = history_df['prediction_correct'].mean() * 100
                st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
                
                # Style the dataframe
                st.dataframe(
                    history_df[['timestamp', 'current_price', 'predicted_price', 'predicted_movement']],
                    use_container_width=True,
                    height=400
                )

if __name__ == "__main__":
    main()
