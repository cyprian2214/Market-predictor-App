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
        self.is_running = False
        self.predictions = {}
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
        
    def fetch_data(self, symbol):
        try:
            st.write(f"Fetching data for {symbol}...")
            # Get data from yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1d', interval='1m')
            
            if df.empty:
                st.error(f"No data received for {symbol}")
                return None
                
            st.write(f"Received {len(df)} rows for {symbol}")
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def prepare_data(self, df):
        try:
            if df is None or df.empty:
                st.error("No data to prepare")
                return None
                
            st.write("Calculating technical indicators...")
            # Calculate technical indicators
            df['RSI'] = ta.momentum.rsi(df['Close'])
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.bollinger_bands(df['Close'])
            df['EMA'] = ta.trend.ema_indicator(df['Close'])
            df['SMA'] = ta.trend.sma_indicator(df['Close'])
            
            # Forward fill NaN values
            df = df.ffill()
            
            st.write("Technical indicators calculated successfully")
            return df
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None

    def make_prediction(self, df):
        try:
            if df is None or df.empty:
                st.error("No data for prediction")
                return None, None
                
            st.write("Preparing features for prediction...")
            # Prepare features
            X = df[['RSI', 'MACD', 'BB_upper', 'BB_lower', 'EMA', 'SMA']].values
            y = df['Close'].values
            
            # Train on most recent data
            train_size = int(len(df) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            st.write("Training model...")
            # Create and train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make prediction
            latest_features = X[-1].reshape(1, -1)
            predicted_price = model.predict(latest_features)[0]
            current_price = df['Close'].iloc[-1]
            
            st.write(f"Prediction complete: Current ${current_price:.2f}, Predicted ${predicted_price:.2f}")
            return predicted_price, current_price
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

    def update_predictions(self, selected_symbols):
        """Update predictions for all selected symbols"""
        try:
            for symbol_name in selected_symbols:
                symbol = self.symbols[symbol_name]
                
                # Fetch and prepare data
                df = self.fetch_data(symbol)
                if df is not None:
                    df = self.prepare_data(df)
                
                if df is not None:
                    # Make prediction
                    predicted_price, current_price = self.make_prediction(df)
                    
                    if predicted_price is not None and current_price is not None:
                        # Store prediction
                        prediction = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': symbol_name,
                            'current_price': current_price,
                            'predicted_price': predicted_price,
                            'predicted_movement': 'Up' if predicted_price > current_price else 'Down',
                            'df': df
                        }
                        
                        if symbol not in self.predictions:
                            self.predictions[symbol] = []
                        self.predictions[symbol].append(prediction)
                        
                        # Keep only last 100 predictions
                        self.predictions[symbol] = self.predictions[symbol][-100:]
            
            return True
        except Exception as e:
            return False

    def create_technical_chart(self, df, symbol_name, predicted_price=None, current_price=None):
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
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA'],
                                line=dict(color='blue', width=1),
                                name='EMA'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA'],
                                line=dict(color='orange', width=1),
                                name='SMA'),
                     row=1, col=1)

        # Add MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                                line=dict(color='blue', width=1),
                                name='MACD'),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'].rolling(window=9).mean(),
                                line=dict(color='orange', width=1),
                                name='Signal'),
                     row=2, col=1)

        # Add prediction arrow if available
        if predicted_price is not None and current_price is not None:
            last_index = df.index[-1]
            next_index = last_index + pd.Timedelta(minutes=1)
            
            arrow_color = 'green' if predicted_price > current_price else 'red'
            arrow_symbol = '⬆' if predicted_price > current_price else '⬇'
            
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
                ay=-40 if predicted_price > current_price else 40
            )
            
            # Add predicted price line
            fig.add_trace(go.Scatter(
                x=[last_index, next_index],
                y=[current_price, predicted_price],
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

def main():
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Market Prediction Pro</h1>
            <p style='font-size: 1.2em; color: #666;'>Advanced Technical Analysis & Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = MarketPredictionApp()
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'update_time' not in st.session_state:
        st.session_state.update_time = time.time()
        
    app = st.session_state.app
    
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
            if st.button("▶ Start", type="primary", disabled=st.session_state.is_running):
                st.session_state.is_running = True
                app.is_running = True
                st.success("✅ Started monitoring markets")
                
        with stop_col:
            if st.button("⏹ Stop", type="secondary", disabled=not st.session_state.is_running):
                st.session_state.is_running = False
                app.is_running = False
                st.info("🛑 Stopped monitoring markets")
    
    # Main content
    if not selected_symbols:
        st.warning("🎯 Please select at least one market to monitor from the sidebar")
        st.markdown("""
        ### Getting Started Guide:
        1. Select markets to monitor from the sidebar:
           - Choose from Major Indices (e.g., NASDAQ-100)
           - Select Tech Stocks (e.g., Apple)
           - Add Forex Pairs if interested
        2. Click the "▶ Start" button to begin monitoring
        3. Wait a few moments for the first predictions to appear
        4. View real-time predictions and technical analysis in the tabs below
        """)
        return
    
    # Update predictions if running
    if st.session_state.is_running:
        current_time = time.time()
        if current_time - st.session_state.update_time >= 60:  # Update every minute
            with st.spinner("Updating predictions..."):
                if app.update_predictions(selected_symbols):
                    st.session_state.update_time = current_time
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Live Analysis", "📊 Historical Data"])
    
    with tab1:
        if st.session_state.is_running and not app.predictions:
            st.info("⏳ Gathering initial data and making predictions...")
        
        # Display predictions in a grid
        for symbol_name in selected_symbols:
            symbol = app.symbols[symbol_name]
            if symbol in app.predictions and app.predictions[symbol]:
                with st.container():
                    st.markdown(f"### {symbol_name}")
                    
                    # Get latest prediction
                    latest = app.predictions[symbol][-1]
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${latest['current_price']:.2f}")
                    with col2:
                        st.metric("Predicted Price", f"${latest['predicted_price']:.2f}")
                    with col3:
                        st.metric("Movement", latest['predicted_movement'])
                    with col4:
                        confidence = abs(latest['predicted_price'] - latest['current_price']) / latest['current_price'] * 100
                        st.metric("Signal Strength", f"{confidence:.1f}%")
                    
                    # Create and display technical chart
                    fig = app.create_technical_chart(latest['df'], symbol_name, latest['predicted_price'], latest['current_price'])
                    st.plotly_chart(fig, use_container_width=True)
            elif st.session_state.is_running:
                st.info(f"⏳ Gathering data for {symbol_name}...")
    
    with tab2:
        if not app.predictions:
            st.info("Start monitoring to see historical prediction data")
        else:
            for symbol_name in selected_symbols:
                symbol = app.symbols[symbol_name]
                if symbol in app.predictions and app.predictions[symbol]:
                    st.subheader(f"{symbol_name} - Historical Predictions")
                    history_df = pd.DataFrame([p for p in app.predictions[symbol] if 'df' not in p])
                    st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main()
