import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Bitcoin AI Prediction Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #f7931a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .sell-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .api-online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .api-offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">₿ Bitcoin AI Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## 🎛️ Configuration")
st.sidebar.markdown("---")

# API Configuration
api_url = st.sidebar.text_input("🔗 API URL", value="https://btc-price-prediction-981m.onrender.com", help="URL of your FastAPI server")

# Data period selection
data_period = st.sidebar.selectbox(
    "📅 Historical Data Period",
    ["30d", "90d", "180d", "1y", "2y"],
    index=2,
    help="Amount of historical data to display"
)

# Prediction settings
st.sidebar.markdown("### 🔮 Prediction Settings")
days_to_predict = st.sidebar.slider("📈 Days to Predict", min_value=1, max_value=30, value=7)
model_type = st.sidebar.selectbox(
    "🤖 AI Model",
    ["auto", "lstm", "linear"],
    help="auto: tries best available model"
)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
st.sidebar.markdown("---")

# Cache functions for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_bitcoin_data(period):
    """Fetch Bitcoin data with caching"""
    try:
        df = yf.download("BTC-USD", period=period, interval="1d", progress=False)
        if df.empty:
            raise ValueError("No data received")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch Bitcoin data: {e}")
        return None

def check_api_status(api_url):
    """Check if API is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def get_current_price(api_url):
    """Get current Bitcoin price from API"""
    try:
        response = requests.get(f"{api_url}/current-price", timeout=5)
        if response.status_code == 200:
            return response.json()["current_price"]
        else:
            return None
    except:
        return None

def make_prediction(api_url, days, model_type):
    """Make prediction via API"""
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"days": days, "model_type": model_type},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": f"Connection error: {str(e)}"}

def create_technical_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    return df

def safe_format_price(price):
    """Safely format price value"""
    try:
        if pd.isna(price):
            return "N/A"
        return f"${float(price):,.2f}"
    except (TypeError, ValueError):
        return "N/A"

def safe_format_percentage(value):
    """Safely format percentage value"""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "N/A"

def safe_format_number(value, decimals=0):
    """Safely format numeric value"""
    try:
        if pd.isna(value):
            return "N/A"
        if decimals == 0:
            return f"{float(value):,.0f}"
        else:
            return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"

# Main dashboard
def main_dashboard():
    # Check API status
    api_online, api_status = check_api_status(api_url)
    
    # Display API status
    if api_online:
        st.markdown(
            '<div class="api-status api-online">✅ API Status: Online</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="api-status api-offline">❌ API Status: Offline - ' + 
            str(api_status.get("error", "Unknown error")) + '</div>',
            unsafe_allow_html=True
        )
    
    # Fetch Bitcoin data
    with st.spinner("📊 Loading Bitcoin data..."):
        df = fetch_bitcoin_data(data_period)
    
    if df is None:
        st.error("Failed to load Bitcoin data. Please check your internet connection.")
        return
    
    if len(df) == 0:
        st.error("No Bitcoin data available.")
        return
    
    # Calculate technical indicators
    df_tech = create_technical_indicators(df)
    
    # Safely extract current price and metrics
    try:
        current_price = float(df['Close'].iloc[-1])
        if len(df) > 1:
            prev_price = float(df['Close'].iloc[-2])
            price_change_24h = ((current_price - prev_price) / prev_price) * 100
        else:
            price_change_24h = 0.0
        
        volume_24h = float(df['Volume'].iloc[-1])
    except (IndexError, TypeError, ValueError) as e:
        st.error(f"Error processing Bitcoin data: {e}")
        return
    
    # Get API current price if available
    api_price = get_current_price(api_url) if api_online else None
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Current Price",
            safe_format_price(current_price),
            safe_format_percentage(price_change_24h)
        )
    
    with col2:
        if api_price:
            try:
                price_diff = float(api_price) - current_price
                st.metric(
                    "🤖 API Price",
                    safe_format_price(api_price),
                    safe_format_price(price_diff)
                )
            except (TypeError, ValueError):
                st.metric("🤖 API Price", "Error", "N/A")
        else:
            st.metric("🤖 API Price", "N/A", "Offline")
    
    with col3:
        st.metric(
            "📊 24h Volume",
            safe_format_number(volume_24h),
            help="Trading volume in the last 24 hours"
        )
    
    with col4:
        try:
            if 'RSI' in df_tech.columns and len(df_tech) > 0:
                current_rsi = float(df_tech['RSI'].iloc[-1])
                if pd.notna(current_rsi):
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric(
                        "📈 RSI",
                        f"{current_rsi:.1f}",
                        rsi_signal
                    )
                else:
                    st.metric("📈 RSI", "N/A", "Calculating...")
            else:
                st.metric("📈 RSI", "N/A", "Calculating...")
        except (IndexError, TypeError, ValueError):
            st.metric("📈 RSI", "Error", "N/A")
    
    # Chart tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🔮 Predictions"])
    
    with tab1:
        st.subheader("📈 Bitcoin Price History")
        
        try:
            # Main price chart
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="BTC Price",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
            
            # Add moving averages if available
            if 'MA21' in df_tech.columns:
                ma21_data = df_tech['MA21'].dropna()
                if len(ma21_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=df_tech['Date'][-len(ma21_data):],
                        y=ma21_data,
                        name='MA21',
                        line=dict(color='blue', width=1)
                    ))
            
            if 'MA50' in df_tech.columns:
                ma50_data = df_tech['MA50'].dropna()
                if len(ma50_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=df_tech['Date'][-len(ma50_data):],
                        y=ma50_data,
                        name='MA50',
                        line=dict(color='red', width=1)
                    ))
            
            fig.update_layout(
                title="Bitcoin Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig_volume.update_layout(
                title="Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating price chart: {e}")
    
    with tab2:
        st.subheader("📊 Technical Indicators")
        
        try:
            if 'RSI' in df_tech.columns and 'MACD' in df_tech.columns:
                # Filter out NaN values
                rsi_data = df_tech[['Date', 'RSI']].dropna()
                macd_data = df_tech[['Date', 'MACD', 'MACD_Signal']].dropna()
                
                if len(rsi_data) > 0 and len(macd_data) > 0:
                    # Create subplots for technical indicators
                    fig_tech = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('RSI (Relative Strength Index)', 'MACD'),
                        vertical_spacing=0.1
                    )
                    
                    # RSI
                    fig_tech.add_trace(
                        go.Scatter(x=rsi_data['Date'], y=rsi_data['RSI'], 
                                 name='RSI', line=dict(color='purple')),
                        row=1, col=1
                    )
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                    
                    # MACD
                    fig_tech.add_trace(
                        go.Scatter(x=macd_data['Date'], y=macd_data['MACD'], 
                                 name='MACD', line=dict(color='blue')),
                        row=2, col=1
                    )
                    if 'MACD_Signal' in macd_data.columns:
                        fig_tech.add_trace(
                            go.Scatter(x=macd_data['Date'], y=macd_data['MACD_Signal'], 
                                     name='Signal', line=dict(color='red')),
                            row=2, col=1
                        )
                    
                    fig_tech.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig_tech, use_container_width=True)
                else:
                    st.info("Not enough data for technical indicators.")
            else:
                st.info("Technical indicators are being calculated...")
        
        except Exception as e:
            st.error(f"Error creating technical indicators: {e}")
    
    with tab3:
        st.subheader("🔮 AI Price Predictions")
        
        if not api_online:
            st.error("🚫 API is offline. Cannot make predictions.")
            st.info("💡 Make sure your FastAPI server is running at the specified URL.")
            return
        
        # Prediction interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Prediction Settings:**")
            st.write(f"• Days to predict: {days_to_predict}")
            st.write(f"• AI Model: {model_type}")
            st.write(f"• API URL: {api_url}")
        
        with col2:
            if st.button("🚀 Make Prediction", type="primary"):
                with st.spinner("🤖 AI is analyzing Bitcoin patterns..."):
                    success, result = make_prediction(api_url, days_to_predict, model_type)
                
                if success:
                    st.session_state['prediction_result'] = result
                    st.success("✅ Prediction completed!")
                else:
                    st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        
        # Display predictions if available
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            if result.get('success', False):
                predictions = result.get('predictions', [])
                metadata = result.get('metadata', {})
                
                if predictions:
                    # Display prediction summary
                    st.markdown("### 📊 Prediction Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        method_used = metadata.get('method_used', 'unknown')
                        st.metric("🎯 Method Used", method_used.upper())
                    with col2:
                        try:
                            confidences = [p.get('confidence', 0) for p in predictions if p.get('confidence')]
                            if confidences:
                                avg_confidence = np.mean(confidences)
                                st.metric("🔮 Avg Confidence", f"{avg_confidence:.1%}")
                            else:
                                st.metric("🔮 Avg Confidence", "N/A")
                        except:
                            st.metric("🔮 Avg Confidence", "N/A")
                    with col3:
                        try:
                            first_pred = float(predictions[0]['predicted_close'])
                            trend = "📈" if first_pred > current_price else "📉"
                            st.metric("📈 Initial Trend", trend)
                        except:
                            st.metric("📈 Initial Trend", "N/A")
                    with col4:
                        try:
                            last_pred = float(predictions[-1]['predicted_close'])
                            total_change = ((last_pred - current_price) / current_price) * 100
                            st.metric("📊 Total Change", f"{total_change:+.2f}%")
                        except:
                            st.metric("📊 Total Change", "N/A")
                    
                    # Predictions table
                    st.markdown("### 📅 Daily Predictions")
                    try:
                        pred_df = pd.DataFrame(predictions)
                        
                        # Safely format the dataframe
                        if 'predicted_close' in pred_df.columns:
                            pred_df['predicted_close'] = pred_df['predicted_close'].apply(
                                lambda x: safe_format_price(x)
                            )
                        if 'confidence' in pred_df.columns:
                            pred_df['confidence'] = pred_df['confidence'].apply(
                                lambda x: f"{float(x):.1%}" if pd.notna(x) and x is not None else "N/A"
                            )
                        if 'trend' in pred_df.columns:
                            pred_df['trend'] = pred_df['trend'].apply(
                                lambda x: {"up": "📈", "down": "📉", "sideways": "➡️"}.get(str(x), str(x))
                            )
                        
                        st.dataframe(
                            pred_df.rename(columns={
                                'date': 'Date',
                                'predicted_close': 'Predicted Price',
                                'confidence': 'Confidence',
                                'trend': 'Trend'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    except Exception as e:
                        st.error(f"Error displaying predictions table: {e}")
                    
                    # Prediction chart
                    st.markdown("### 📈 Prediction Visualization")
                    
                    try:
                        # Combine historical and predicted data
                        historical_dates = df['Date'].tail(30)  # Last 30 days
                        historical_prices = df['Close'].tail(30)
                        
                        pred_dates = pd.to_datetime([p['date'] for p in predictions])
                        pred_prices = [float(p['predicted_close']) for p in predictions]
                        
                        fig_pred = go.Figure()
                        
                        # Historical data
                        fig_pred.add_trace(go.Scatter(
                            x=historical_dates,
                            y=historical_prices,
                            name='Historical Price',
                            line=dict(color='#f7931a', width=2)
                        ))
                        
                        # Predicted data
                        fig_pred.add_trace(go.Scatter(
                            x=pred_dates,
                            y=pred_prices,
                            name='Predicted Price',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        # Add vertical line at prediction start
                        fig_pred.add_vline(
                            x=df['Date'].iloc[-1],
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Prediction Start"
                        )
                        
                        fig_pred.update_layout(
                            title=f"Bitcoin Price Prediction ({days_to_predict} days)",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error creating prediction chart: {e}")
                    
                    # Model metadata
                    with st.expander("🔍 Model Details"):
                        st.json(metadata)
                else:
                    st.warning("No predictions received from API.")
            else:
                st.error(f"Prediction failed: {result.get('message', 'Unknown error')}")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Run main dashboard
if __name__ == "__main__":
    try:
        main_dashboard()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.info("Please refresh the page and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Cryptocurrency trading involves high risk. 
    Always do your own research and never invest more than you can afford to lose.</p>
""", unsafe_allow_html=True)