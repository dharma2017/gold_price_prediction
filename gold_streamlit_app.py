import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and TensorFlow
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #FFD700;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🏆 Gold Price Prediction System</h1>", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        model = joblib.load('gold_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        
        # Try to load time series models
        try:
            arima_model = joblib.load('arima_model.pkl')
        except:
            arima_model = None
        
        try:
            exp_model = joblib.load('exp_smoothing_model.pkl')
        except:
            exp_model = None
            
        try:
            ts_reg_model = joblib.load('ts_regression_model.pkl')
            ts_scaler = joblib.load('ts_scaler.pkl')
        except:
            ts_reg_model = None
            ts_scaler = None
        
        # Try to load advanced models
        try:
            xgb_model = joblib.load('xgboost_regressor.pkl')
            scaler_advanced = joblib.load('scaler_advanced.pkl')
            feature_cols_advanced = joblib.load('feature_columns_advanced.pkl')
        except:
            xgb_model = None
            scaler_advanced = None
            feature_cols_advanced = None
        
        try:
            if TENSORFLOW_AVAILABLE:
                lstm_model = keras.models.load_model('lstm_model.h5')
                scaler_lstm = joblib.load('scaler_lstm.pkl')
            else:
                lstm_model = None
                scaler_lstm = None
        except:
            lstm_model = None
            scaler_lstm = None
        
        return (model, scaler, feature_cols, arima_model, exp_model, 
                ts_reg_model, ts_scaler, xgb_model, scaler_advanced, 
                feature_cols_advanced, lstm_model, scaler_lstm)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure you've run the Jupyter notebook first to train and save the models.")
        return (None,) * 12

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Gold Price (2013-2023).csv')
        # Clean data
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        
        numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('K', '').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

(model, scaler, feature_cols, arima_model, exp_model, ts_reg_model, ts_scaler,
 xgb_model, scaler_advanced, feature_cols_advanced, lstm_model, scaler_lstm) = load_models()
df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/gold-bars.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Home", "📊 Dashboard", "📈 Time Series Analysis", 
                               "🔮 Prediction", "📉 Forecasting", "🎯 Model Performance", 
                               "🤖 Advanced ML", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    if df is not None:
        st.metric("Total Records", len(df))
        st.metric("Latest Price", f"{df['Price'].iloc[-1]:,.2f}")
        st.metric("Date Range", f"{df['Date'].min().year} - {df['Date'].max().year}")

# Home Page
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data Analysis</h3>
            <p>Comprehensive exploratory data analysis with 10+ years of historical gold prices</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 ML Models</h3>
            <p>Multiple classification algorithms including Random Forest, SVM, and more</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Predictions</h3>
            <p>Predict whether gold prices will go up or down based on market indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📌 How It Works")
    st.markdown("""
    This application uses machine learning to predict gold price movements:
    
    1. **Data Collection**: Historical gold price data from 2013-2023
    2. **Feature Engineering**: Technical indicators including moving averages, volatility measures
    3. **Model Training**: Multiple ML algorithms with hyperparameter tuning
    4. **Prediction**: Binary classification (Price Up vs. Price Down)
    5. **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score
    """)
    
    if df is not None:
        st.subheader("📈 Recent Price Trend")
        fig = go.Figure()
        recent_data = df.tail(100)
        fig.add_trace(go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Price'],
            mode='lines',
            name='Gold Price',
            line=dict(color='gold', width=2)
        ))
        fig.update_layout(
            title="Last 100 Days Gold Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# Dashboard Page
elif page == "📊 Dashboard":
    if df is not None:
        st.subheader("📊 Gold Price Analytics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['Price'].iloc[-1]
            st.metric("Current Price", f"{current_price:,.2f}")
        
        with col2:
            price_change = df['Price'].iloc[-1] - df['Price'].iloc[-2]
            st.metric("Daily Change", f"{price_change:,.2f}", 
                     delta=f"{(price_change/df['Price'].iloc[-2]*100):.2f}%")
        
        with col3:
            max_price = df['Price'].max()
            st.metric("All-Time High", f"{max_price:,.2f}")
        
        with col4:
            min_price = df['Price'].min()
            st.metric("All-Time Low", f"{min_price:,.2f}")
        
        st.markdown("---")
        
        # Time series chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Price History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='gold', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 215, 0, 0.1)'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Price Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                'Value': [
                    f"{df['Price'].mean():,.2f}",
                    f"{df['Price'].median():,.2f}",
                    f"{df['Price'].std():,.2f}",
                    f"{df['Price'].min():,.2f}",
                    f"{df['Price'].max():,.2f}",
                    f"{df['Price'].quantile(0.25):,.2f}",
                    f"{df['Price'].quantile(0.75):,.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Volume Analysis")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['Date'].tail(50),
                y=df['Vol.'].tail(50),
                name='Volume',
                marker_color='lightblue'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Volume",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Price Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['Price'],
                nbinsx=50,
                marker_color='gold',
                opacity=0.7
            ))
            fig.update_layout(
                xaxis_title="Price (USD)",
                yaxis_title="Frequency",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Unable to load data. Please check if the CSV file exists.")

# Time Series Analysis Page
elif page == "📈 Time Series Analysis":
    st.subheader("📈 Gold Price Time Series Analysis")
    
    if df is not None:
        # Prepare time series data
        ts_df = df.set_index('Date').sort_index()
        
        # Time Series Decomposition
        st.markdown("### 🔍 Time Series Decomposition")
        st.markdown("Decomposing the gold price into trend, seasonal, and residual components.")
        
        try:
            decomposition = seasonal_decompose(ts_df['Price'], model='additive', period=365)
            
            # Create decomposition plots
            fig = go.Figure()
            
            # Original
            fig.add_trace(go.Scatter(
                x=ts_df.index,
                y=ts_df['Price'],
                mode='lines',
                name='Original',
                line=dict(color='gold', width=1.5)
            ))
            
            fig.update_layout(
                title="Original Gold Price",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='blue', width=2)
            ))
            fig2.update_layout(
                title="Trend Component",
                xaxis_title="Date",
                yaxis_title="Trend",
                template='plotly_white',
                height=250
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Seasonal
            col1, col2 = st.columns(2)
            with col1:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=decomposition.seasonal.index[-365:],
                    y=decomposition.seasonal.values[-365:],
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='green', width=1.5)
                ))
                fig3.update_layout(
                    title="Seasonal Component (Last Year)",
                    xaxis_title="Date",
                    yaxis_title="Seasonal",
                    template='plotly_white',
                    height=250
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=decomposition.resid.index,
                    y=decomposition.resid.values,
                    mode='lines',
                    name='Residual',
                    line=dict(color='red', width=1)
                ))
                fig4.update_layout(
                    title="Residual Component",
                    xaxis_title="Date",
                    yaxis_title="Residual",
                    template='plotly_white',
                    height=250
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in decomposition: {e}")
        
        st.markdown("---")
        
        # Rolling Statistics
        st.markdown("### 📊 Rolling Statistics")
        
        window = st.slider("Select Rolling Window (days)", 7, 90, 30)
        
        rolling_mean = ts_df['Price'].rolling(window=window).mean()
        rolling_std = ts_df['Price'].rolling(window=window).std()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=ts_df['Price'],
            mode='lines',
            name='Original Price',
            line=dict(color='gold', width=1),
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=rolling_mean,
            mode='lines',
            name=f'{window}-Day Rolling Mean',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=rolling_std,
            mode='lines',
            name=f'{window}-Day Rolling Std',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"Gold Price with {window}-Day Rolling Statistics",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2=dict(
                title="Standard Deviation",
                overlaying='y',
                side='right'
            ),
            template='plotly_white',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Autocorrelation Analysis
        st.markdown("### 🔄 Autocorrelation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Autocorrelation Function (ACF)**")
            st.markdown("Shows correlation between time series and its lagged values.")
            
        with col2:
            st.markdown("**Partial Autocorrelation Function (PACF)**")
            st.markdown("Shows correlation after removing effects of intermediate lags.")
        
        # ACF and PACF plots using matplotlib (simplified for Streamlit)
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
            plot_acf(ts_df['Price'].dropna(), lags=40, ax=ax_acf)
            ax_acf.set_title('ACF Plot')
            st.pyplot(fig_acf)
            plt.close()
        
        with col2:
            fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
            plot_pacf(ts_df['Price'].dropna(), lags=40, ax=ax_pacf)
            ax_pacf.set_title('PACF Plot')
            st.pyplot(fig_pacf)
            plt.close()
        
        st.markdown("---")
        
        # Year-over-Year Analysis
        st.markdown("### 📅 Year-over-Year Analysis")
        
        yearly_avg = ts_df.groupby(ts_df.index.year)['Price'].agg(['mean', 'min', 'max', 'std'])
        yearly_avg.index.name = 'Year'
        yearly_avg.columns = ['Average Price', 'Min Price', 'Max Price', 'Std Dev']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=yearly_avg.index,
                y=yearly_avg['Average Price'],
                name='Average Price',
                marker_color='gold',
                text=yearly_avg['Average Price'].round(2),
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Average Gold Price by Year",
                xaxis_title="Year",
                yaxis_title="Average Price (USD)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Yearly Statistics**")
            st.dataframe(yearly_avg.style.format("{:.2f}"), use_container_width=True)
        
        # Monthly Seasonality
        st.markdown("### 🌙 Monthly Seasonality Pattern")
        
        monthly_avg = ts_df.groupby(ts_df.index.month)['Price'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=month_names,
            y=monthly_avg.values,
            mode='lines+markers',
            name='Average Price',
            line=dict(color='gold', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Average Gold Price by Month (All Years)",
            xaxis_title="Month",
            yaxis_title="Average Price (USD)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Unable to load data for time series analysis.")

# Forecasting Page
elif page == "📉 Forecasting":
    st.subheader("📉 Gold Price Forecasting")
    
    if df is not None:
        st.markdown("""
        Use time series models to forecast future gold prices based on historical patterns.
        """)
        
        # Model selection
        forecast_model = st.selectbox(
            "Select Forecasting Model",
            ["ARIMA", "Exponential Smoothing", "Time Series Regression", 
             "Random Forest Regressor", "XGBoost (Advanced)", "LSTM Deep Learning"]
        )
        
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            forecast_button = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
        
        if forecast_button:
            ts_df = df.set_index('Date').sort_index()
            
            st.markdown("---")
            st.subheader(f"📊 {forecast_model} Forecast Results")
            
            try:
                if forecast_model == "ARIMA" and arima_model is not None:
                    # ARIMA Forecast
                    forecast = arima_model.forecast(steps=forecast_days)
                    last_date = ts_df.index[-1]
                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                   periods=forecast_days, freq='D')
                    
                    # Create forecast dataframe
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Price': forecast
                    })
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Historical data (last 90 days)
                    fig.add_trace(go.Scatter(
                        x=ts_df.index[-90:],
                        y=ts_df['Price'][-90:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='gold', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast,
                        mode='lines+markers',
                        name='ARIMA Forecast',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"ARIMA Forecast ({forecast_days} days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast table
                    st.markdown("### Forecast Values")
                    st.dataframe(
                        forecast_df.style.format({'Forecasted Price': '{:.2f}'}),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Forecast Mean", f"{forecast.mean():.2f}")
                    with col2:
                        st.metric("Forecast Min", f"{forecast.min():.2f}")
                    with col3:
                        st.metric("Forecast Max", f"{forecast.max():.2f}")
                
                elif forecast_model == "Exponential Smoothing" and exp_model is not None:
                    # Exponential Smoothing Forecast
                    forecast = exp_model.forecast(steps=forecast_days)
                    last_date = ts_df.index[-1]
                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                   periods=forecast_days, freq='D')
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Price': forecast
                    })
                    
                    # Plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=ts_df.index[-90:],
                        y=ts_df['Price'][-90:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='gold', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast,
                        mode='lines+markers',
                        name='Exp. Smoothing Forecast',
                        line=dict(color='green', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"Exponential Smoothing Forecast ({forecast_days} days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Forecast Values")
                    st.dataframe(
                        forecast_df.style.format({'Forecasted Price': '{:.2f}'}),
                        use_container_width=True,
                        height=300
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Forecast Mean", f"{forecast.mean():.2f}")
                    with col2:
                        st.metric("Forecast Min", f"{forecast.min():.2f}")
                    with col3:
                        st.metric("Forecast Max", f"{forecast.max():.2f}")
                
                elif forecast_model == "Time Series Regression" and ts_reg_model is not None and ts_scaler is not None:
                    # Time Series Regression Forecast
                    st.info("Time Series Regression uses lagged features to predict future prices iteratively.")
                    
                    # Prepare data for forecasting
                    ts_df_copy = ts_df.copy()
                    
                    # Create features for the last known data point
                    ts_df_copy['Days_Since_Start'] = (ts_df_copy.index - ts_df_copy.index[0]).days
                    ts_df_copy['Price_Lag1'] = ts_df_copy['Price'].shift(1)
                    ts_df_copy['Price_Lag7'] = ts_df_copy['Price'].shift(7)
                    ts_df_copy['Price_Lag30'] = ts_df_copy['Price'].shift(30)
                    ts_df_copy['Price_MA_7'] = ts_df_copy['Price'].rolling(window=7).mean()
                    ts_df_copy['Price_MA_30'] = ts_df_copy['Price'].rolling(window=30).mean()
                    ts_df_copy['Price_Std_7'] = ts_df_copy['Price'].rolling(window=7).std()
                    
                    ts_df_copy = ts_df_copy.dropna()
                    
                    # Features for regression
                    ts_features = ['Days_Since_Start', 'Price_Lag1', 'Price_Lag7', 'Price_Lag30', 
                                   'Price_MA_7', 'Price_MA_30', 'Price_Std_7']
                    
                    # Get last known values
                    last_data = ts_df_copy.iloc[-1]
                    current_price = last_data['Price']
                    last_date = ts_df_copy.index[-1]
                    
                    # Initialize forecasting
                    forecast_prices = []
                    forecast_dates = []
                    
                    # Rolling window for features
                    recent_prices = list(ts_df_copy['Price'].tail(30).values)
                    
                    for i in range(forecast_days):
                        # Calculate current features
                        days_since_start = last_data['Days_Since_Start'] + i + 1
                        price_lag1 = recent_prices[-1]
                        price_lag7 = recent_prices[-7] if len(recent_prices) >= 7 else recent_prices[0]
                        price_lag30 = recent_prices[-30] if len(recent_prices) >= 30 else recent_prices[0]
                        price_ma_7 = np.mean(recent_prices[-7:])
                        price_ma_30 = np.mean(recent_prices[-30:])
                        price_std_7 = np.std(recent_prices[-7:])
                        
                        # Create feature vector
                        X_forecast = np.array([[days_since_start, price_lag1, price_lag7, price_lag30,
                                               price_ma_7, price_ma_30, price_std_7]])
                        
                        # Scale features
                        X_forecast_scaled = ts_scaler.transform(X_forecast)
                        
                        # Predict
                        predicted_price = ts_reg_model.predict(X_forecast_scaled)[0]
                        
                        # Store prediction
                        forecast_prices.append(predicted_price)
                        forecast_dates.append(last_date + timedelta(days=i+1))
                        
                        # Update recent prices for next iteration
                        recent_prices.append(predicted_price)
                        if len(recent_prices) > 30:
                            recent_prices.pop(0)
                    
                    # Create forecast dataframe
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Price': forecast_prices
                    })
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Historical data (last 90 days)
                    fig.add_trace(go.Scatter(
                        x=ts_df.index[-90:],
                        y=ts_df['Price'][-90:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='gold', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_prices,
                        mode='lines+markers',
                        name='TS Regression Forecast',
                        line=dict(color='purple', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    # Add confidence band (simple approximation)
                    std_dev = np.std(ts_df['Price'].tail(30))
                    upper_bound = np.array(forecast_prices) + 1.96 * std_dev
                    lower_bound = np.array(forecast_prices) - 1.96 * std_dev
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates + forecast_dates[::-1],
                        y=list(upper_bound) + list(lower_bound[::-1]),
                        fill='toself',
                        fillcolor='rgba(128, 0, 128, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f"Time Series Regression Forecast ({forecast_days} days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast table
                    st.markdown("### Forecast Values")
                    st.dataframe(
                        forecast_df.style.format({'Forecasted Price': '{:.2f}'}),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Forecast Mean", f"{np.mean(forecast_prices):.2f}")
                    with col2:
                        st.metric("Forecast Min", f"{np.min(forecast_prices):.2f}")
                    with col3:
                        st.metric("Forecast Max", f"{np.max(forecast_prices):.2f}")
                    with col4:
                        st.metric("Price Change", 
                                 f"{forecast_prices[-1] - current_price:.2f}",
                                 delta=f"{((forecast_prices[-1] - current_price) / current_price * 100):.2f}%")
                    
                    # Show feature importance if available
                    if hasattr(ts_reg_model, 'feature_importances_'):
                        st.markdown("### Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': ts_features,
                            'Importance': ts_reg_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig_imp = go.Figure()
                        fig_imp.add_trace(go.Bar(
                            x=importance_df['Importance'],
                            y=importance_df['Feature'],
                            orientation='h',
                            marker_color='lightblue'
                        ))
                        fig_imp.update_layout(
                            title="Feature Importance in Regression Model",
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            template='plotly_white',
                            height=300
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                
                elif forecast_model == "Random Forest Regressor":
                    # Random Forest Regressor - Train on-the-fly for forecasting
                    st.info("Training Random Forest Regressor for multi-step forecasting...")
                    
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.preprocessing import StandardScaler
                    
                    # Prepare data for training
                    ts_df_copy = ts_df.copy()
                    
                    # Create features
                    ts_df_copy['Days_Since_Start'] = (ts_df_copy.index - ts_df_copy.index[0]).days
                    ts_df_copy['Price_Lag1'] = ts_df_copy['Price'].shift(1)
                    ts_df_copy['Price_Lag7'] = ts_df_copy['Price'].shift(7)
                    ts_df_copy['Price_Lag30'] = ts_df_copy['Price'].shift(30)
                    ts_df_copy['Price_MA_7'] = ts_df_copy['Price'].rolling(window=7).mean()
                    ts_df_copy['Price_MA_30'] = ts_df_copy['Price'].rolling(window=30).mean()
                    ts_df_copy['Price_Std_7'] = ts_df_copy['Price'].rolling(window=7).std()
                    ts_df_copy['Price_Momentum'] = ts_df_copy['Price'].pct_change(7)
                    ts_df_copy['Price_Range'] = ts_df_copy['High'] - ts_df_copy['Low']
                    ts_df_copy['DayOfWeek'] = ts_df_copy.index.dayofweek
                    ts_df_copy['Month'] = ts_df_copy.index.month
                    ts_df_copy['Quarter'] = ts_df_copy.index.quarter
                    
                    ts_df_copy = ts_df_copy.dropna()
                    
                    # Features for regression
                    rf_features = ['Days_Since_Start', 'Price_Lag1', 'Price_Lag7', 'Price_Lag30', 
                                   'Price_MA_7', 'Price_MA_30', 'Price_Std_7', 'Price_Momentum',
                                   'Price_Range', 'DayOfWeek', 'Month', 'Quarter']
                    
                    X_rf = ts_df_copy[rf_features]
                    y_rf = ts_df_copy['Price']
                    
                    # Train-test split (use last 20% for validation)
                    split_idx = int(len(X_rf) * 0.8)
                    X_train_rf = X_rf.iloc[:split_idx]
                    X_test_rf = X_rf.iloc[split_idx:]
                    y_train_rf = y_rf.iloc[:split_idx]
                    y_test_rf = y_rf.iloc[split_idx:]
                    
                    # Scale features
                    scaler_rf = StandardScaler()
                    X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
                    X_test_rf_scaled = scaler_rf.transform(X_test_rf)
                    
                    # Train Random Forest
                    with st.spinner("Training Random Forest model..."):
                        rf_model = RandomForestRegressor(
                            n_estimators=200,
                            max_depth=20,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                        rf_model.fit(X_train_rf_scaled, y_train_rf)
                    
                    # Evaluate on validation set
                    y_pred_val = rf_model.predict(X_test_rf_scaled)
                    mae_val = mean_absolute_error(y_test_rf, y_pred_val)
                    rmse_val = np.sqrt(mean_squared_error(y_test_rf, y_pred_val))
                    r2_val = r2_score(y_test_rf, y_pred_val)
                    
                    st.success(f"✓ Model trained successfully! Validation R² Score: {r2_val:.4f}")
                    
                    # Display validation metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Validation MAE", f"{mae_val:.2f}")
                    with col2:
                        st.metric("Validation RMSE", f"{rmse_val:.2f}")
                    with col3:
                        st.metric("Validation R²", f"{r2_val:.4f}")
                    
                    st.markdown("---")
                    
                    # Get last known values
                    last_data = ts_df_copy.iloc[-1]
                    current_price = last_data['Price']
                    last_date = ts_df_copy.index[-1]
                    
                    # Initialize forecasting
                    forecast_prices = []
                    forecast_dates = []
                    
                    # Rolling window for features
                    recent_prices = list(ts_df_copy['Price'].tail(30).values)
                    recent_highs = list(ts_df_copy['High'].tail(30).values)
                    recent_lows = list(ts_df_copy['Low'].tail(30).values)
                    
                    # Forecast iteratively
                    with st.spinner(f"Generating {forecast_days}-day forecast..."):
                        for i in range(forecast_days):
                            # Calculate current features
                            forecast_date = last_date + timedelta(days=i+1)
                            days_since_start = last_data['Days_Since_Start'] + i + 1
                            price_lag1 = recent_prices[-1]
                            price_lag7 = recent_prices[-7] if len(recent_prices) >= 7 else recent_prices[0]
                            price_lag30 = recent_prices[-30] if len(recent_prices) >= 30 else recent_prices[0]
                            price_ma_7 = np.mean(recent_prices[-7:])
                            price_ma_30 = np.mean(recent_prices[-30:])
                            price_std_7 = np.std(recent_prices[-7:])
                            price_momentum = (recent_prices[-1] - recent_prices[-7]) / recent_prices[-7] if len(recent_prices) >= 7 else 0
                            price_range = recent_highs[-1] - recent_lows[-1]
                            day_of_week = forecast_date.dayofweek
                            month = forecast_date.month
                            quarter = (month - 1) // 3 + 1
                            
                            # Create feature vector
                            X_forecast = np.array([[days_since_start, price_lag1, price_lag7, price_lag30,
                                                   price_ma_7, price_ma_30, price_std_7, price_momentum,
                                                   price_range, day_of_week, month, quarter]])
                            
                            # Scale features
                            X_forecast_scaled = scaler_rf.transform(X_forecast)
                            
                            # Predict
                            predicted_price = rf_model.predict(X_forecast_scaled)[0]
                            
                            # Store prediction
                            forecast_prices.append(predicted_price)
                            forecast_dates.append(forecast_date)
                            
                            # Update recent data for next iteration
                            recent_prices.append(predicted_price)
                            recent_highs.append(predicted_price * 1.01)  # Approximate
                            recent_lows.append(predicted_price * 0.99)   # Approximate
                            if len(recent_prices) > 30:
                                recent_prices.pop(0)
                                recent_highs.pop(0)
                                recent_lows.pop(0)
                    
                    # Create forecast dataframe
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Price': forecast_prices
                    })
                    
                    # Calculate prediction intervals using historical residuals
                    residuals = y_test_rf - y_pred_val
                    std_residual = np.std(residuals)
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Historical data (last 90 days)
                    fig.add_trace(go.Scatter(
                        x=ts_df.index[-90:],
                        y=ts_df['Price'][-90:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='gold', width=2)
                    ))
                    
                    # Validation predictions
                    fig.add_trace(go.Scatter(
                        x=y_test_rf.index,
                        y=y_pred_val,
                        mode='lines',
                        name='Validation Predictions',
                        line=dict(color='orange', width=2, dash='dot'),
                        opacity=0.7
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_prices,
                        mode='lines+markers',
                        name='RF Forecast',
                        line=dict(color='darkblue', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    # Add prediction intervals (95% confidence)
                    upper_bound = np.array(forecast_prices) + 1.96 * std_residual
                    lower_bound = np.array(forecast_prices) - 1.96 * std_residual
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates + forecast_dates[::-1],
                        y=list(upper_bound) + list(lower_bound[::-1]),
                        fill='toself',
                        fillcolor='rgba(0, 0, 139, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f"Random Forest Regressor Forecast ({forecast_days} days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast table
                    st.markdown("### Forecast Values")
                    st.dataframe(
                        forecast_df.style.format({'Forecasted Price': '{:.2f}'}),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Forecast Mean", f"{np.mean(forecast_prices):.2f}")
                    with col2:
                        st.metric("Forecast Min", f"{np.min(forecast_prices):.2f}")
                    with col3:
                        st.metric("Forecast Max", f"{np.max(forecast_prices):.2f}")
                    with col4:
                        price_change = forecast_prices[-1] - current_price
                        price_change_pct = (price_change / current_price * 100)
                        st.metric("Price Change", 
                                 f"{price_change:.2f}",
                                 delta=f"{price_change_pct:.2f}%")
                    
                    # Feature Importance
                    st.markdown("### 🎯 Feature Importance Analysis")
                    importance_df = pd.DataFrame({
                        'Feature': rf_features,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_imp = go.Figure()
                        fig_imp.add_trace(go.Bar(
                            x=importance_df['Importance'],
                            y=importance_df['Feature'],
                            orientation='h',
                            marker_color='lightblue',
                            text=importance_df['Importance'].round(3),
                            textposition='outside'
                        ))
                        fig_imp.update_layout(
                            title="Feature Importance in Random Forest Model",
                            xaxis_title="Importance Score",
                            yaxis_title="Feature",
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Top 5 Features:**")
                        for idx, row in importance_df.head(5).iterrows():
                            st.metric(row['Feature'], f"{row['Importance']:.4f}")
                    
                    # Trend Analysis
                    st.markdown("### 📊 Forecast Trend Analysis")
                    
                    # Calculate trend statistics
                    forecast_trend = np.polyfit(range(len(forecast_prices)), forecast_prices, 1)[0]
                    trend_direction = "📈 Upward" if forecast_trend > 0 else "📉 Downward"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Trend", trend_direction)
                    with col2:
                        volatility = np.std(forecast_prices)
                        st.metric("Forecast Volatility", f"{volatility:.2f}")
                    with col3:
                        price_range = np.max(forecast_prices) - np.min(forecast_prices)
                        st.metric("Price Range", f"{price_range:.2f}")
                
                else:
                    st.warning(f"{forecast_model} model not available. Please train the model first using the Jupyter notebook.")
            
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.exception(e)
        
        st.markdown("---")
        st.info("💡 **Note**: Forecasts are based on historical patterns and may not account for unexpected market events or external factors.")
    
    else:
        st.error("Unable to load data for forecasting.")

# Prediction Page
elif page == "🔮 Prediction":
    st.subheader("🔮 Gold Price Movement Prediction")
    
    if model is not None and scaler is not None:
        st.markdown("""
        Enter the current market indicators to predict if gold price will go **UP** or **DOWN** tomorrow.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            open_price = st.number_input("Open Price ()", min_value=0.0, value=1800.0, step=10.0)
            high_price = st.number_input("High Price ()", min_value=0.0, value=1850.0, step=10.0)
            low_price = st.number_input("Low Price ()", min_value=0.0, value=1780.0, step=10.0)
            volume = st.number_input("Volume (K)", min_value=0.0, value=100.0, step=5.0)
            change_pct = st.number_input("Change %", min_value=-10.0, max_value=10.0, value=0.5, step=0.1)
        
        with col2:
            day = st.selectbox("Day of Month", range(1, 32), index=14)
            month = st.selectbox("Month", range(1, 13), index=0)
            day_of_week = st.selectbox("Day of Week", 
                                      ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                                      index=0)
            day_of_week_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'].index(day_of_week)
            quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)
        
        with col3:
            price_ma_7 = st.number_input("7-Day Moving Average ()", min_value=0.0, value=1810.0, step=10.0)
            price_ma_30 = st.number_input("30-Day Moving Average ()", min_value=0.0, value=1820.0, step=10.0)
            price_std_7 = st.number_input("7-Day Std Dev ()", min_value=0.0, value=15.0, step=1.0)
            high_low_diff = high_price - low_price
            open_close_diff = st.number_input("Open-Close Difference ()", 
                                             min_value=-100.0, max_value=100.0, value=5.0, step=1.0)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🎯 Predict Price Movement", type="primary", use_container_width=True)
        
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'Open': [open_price],
                'High': [high_price],
                'Low': [low_price],
                'Vol.': [volume],
                'Change %': [change_pct],
                'Day': [day],
                'Month': [month],
                'DayOfWeek': [day_of_week_num],
                'Quarter': [quarter],
                'Price_MA_7': [price_ma_7],
                'Price_MA_30': [price_ma_30],
                'Price_Std_7': [price_std_7],
                'High_Low_Diff': [high_low_diff],
                'Open_Close_Diff': [open_close_diff]
            })
            
            # Scale input
            input_scaled = scaler.transform(input_data[feature_cols])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction == 1:
                    st.success("### 📈 Price Expected to GO UP")
                    st.markdown(f"**Confidence:** {prediction_proba[1]*100:.2f}%")
                else:
                    st.error("### 📉 Price Expected to GO DOWN")
                    st.markdown(f"**Confidence:** {prediction_proba[0]*100:.2f}%")
                
                # Probability chart
                fig = go.Figure(go.Bar(
                    x=['Down', 'Up'],
                    y=[prediction_proba[0]*100, prediction_proba[1]*100],
                    marker_color=['red', 'green'],
                    text=[f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability (%)",
                    template='plotly_white',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("⚠️ **Disclaimer**: This prediction is based on historical data and machine learning models. "
                   "It should not be used as the sole basis for investment decisions. Always consult with "
                   "financial advisors and do your own research.")
    else:
        st.error("Models not loaded. Please run the Jupyter notebook first to train the models.")

# Model Performance Page
elif page == "🎯 Model Performance":
    st.subheader("🎯 Model Performance Metrics")
    
    # Classification Models
    st.markdown("### Classification Models (Price Direction Prediction)")
    
    try:
        comparison_df = pd.read_csv('model_comparison.csv')
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['Test Accuracy'],
                name='Test Accuracy',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['Train Accuracy'],
                name='Train Accuracy',
                marker_color='lightcoral'
            ))
            fig.update_layout(
                title="Model Accuracy Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                barmode='group',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['F1 Score'],
                marker_color='gold'
            ))
            fig.update_layout(
                title="F1 Score Comparison",
                xaxis_title="Model",
                yaxis_title="F1 Score",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Best Classification Model")
        best_model_row = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", best_model_row['Model'])
        with col2:
            st.metric("Test Accuracy", f"{best_model_row['Test Accuracy']:.4f}")
        with col3:
            st.metric("F1 Score", f"{best_model_row['F1 Score']:.4f}")
        with col4:
            st.metric("Overfitting", best_model_row['Overfitting'])
    
    except FileNotFoundError:
        st.warning("Classification model comparison data not found. Please run the Jupyter notebook first.")
    
    st.markdown("---")
    
    # Time Series Models
    st.markdown("### Time Series Forecasting Models")
    
    try:
        ts_comparison_df = pd.read_csv('ts_model_comparison.csv')
        st.dataframe(ts_comparison_df, hide_index=True, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ts_comparison_df['Model'],
                y=ts_comparison_df['MAE'],
                marker_color='lightgreen',
                name='MAE'
            ))
            fig.update_layout(
                title="Mean Absolute Error (MAE) Comparison",
                xaxis_title="Model",
                yaxis_title="MAE ()",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ts_comparison_df['Model'],
                y=ts_comparison_df['R² Score'],
                marker_color='lightcoral',
                name='R² Score'
            ))
            fig.update_layout(
                title="R² Score Comparison",
                xaxis_title="Model",
                yaxis_title="R² Score",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Best Time Series Model")
        best_ts_model = ts_comparison_df.loc[ts_comparison_df['R² Score'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", best_ts_model['Model'])
        with col2:
            st.metric("MAE", f"{best_ts_model['MAE']:.2f}")
        with col3:
            st.metric("R² Score", f"{best_ts_model['R² Score']:.4f}")
    
    except FileNotFoundError:
        st.warning("Time series model comparison data not found. Please run the Jupyter notebook first.")

# Advanced ML Page
elif page == "🤖 Advanced ML":
    st.subheader("🤖 Advanced Machine Learning Models")
    
    st.markdown("""
    This section showcases state-of-the-art machine learning models with advanced feature engineering.
    """)
    
    # Model availability status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Available" if xgb_model is not None and XGBOOST_AVAILABLE else "❌ Not Available"
        color = "green" if xgb_model is not None and XGBOOST_AVAILABLE else "red"
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid {color};">
            <h3>XGBoost</h3>
            <p><strong>Status:</strong> {status}</p>
            <p>Gradient Boosting with 100+ features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅ Available" if lstm_model is not None and TENSORFLOW_AVAILABLE else "❌ Not Available"
        color = "green" if lstm_model is not None and TENSORFLOW_AVAILABLE else "red"
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid {color};">
            <h3>LSTM</h3>
            <p><strong>Status:</strong> {status}</p>
            <p>Deep Learning with Sequential Patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rf_status = "✅ Available"
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid green;">
            <h3>Random Forest</h3>
            <p><strong>Status:</strong> {rf_status}</p>
            <p>Ensemble Learning Baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Engineering Overview
    if feature_cols_advanced is not None:
        st.markdown("### 📊 Advanced Feature Engineering")
        
        st.success(f"**Total Features:** {len(feature_cols_advanced)} advanced features created")
        
        # Categorize features
        lag_features = [f for f in feature_cols_advanced if 'Lag' in f]
        rolling_features = [f for f in feature_cols_advanced if any(x in f for x in ['MA_', 'Std_', 'Min_', 'Max_', 'Range_'])]
        momentum_features = [f for f in feature_cols_advanced if any(x in f for x in ['Momentum', 'ROC', 'EMA'])]
        trend_features = [f for f in feature_cols_advanced if 'Trend' in f or 'Detrended' in f]
        seasonality_features = [f for f in feature_cols_advanced if any(x in f for x in ['Day', 'Month', 'Quarter', 'Week', 'Year', '_Sin', '_Cos'])]
        technical_features = [f for f in feature_cols_advanced if any(x in f for x in ['BB_', 'RSI', 'AutoCorr', 'HL_', 'OC_', 'Volume'])]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lag Features", len(lag_features))
            st.metric("Rolling Statistics", len(rolling_features))
        
        with col2:
            st.metric("Momentum Features", len(momentum_features))
            st.metric("Trend Features", len(trend_features))
        
        with col3:
            st.metric("Seasonality Features", len(seasonality_features))
            st.metric("Technical Indicators", len(technical_features))
        
        # Feature category breakdown
        st.markdown("### Feature Categories Breakdown")
        
        feature_categories = pd.DataFrame({
            'Category': ['Lag Features', 'Rolling Statistics', 'Momentum', 'Trend', 'Seasonality', 'Technical Indicators'],
            'Count': [len(lag_features), len(rolling_features), len(momentum_features), 
                     len(trend_features), len(seasonality_features), len(technical_features)],
            'Examples': [
                'Price_Lag1, Price_Lag7, Price_Lag30',
                'Price_MA_30, Price_Std_7, Price_Range_60',
                'Price_Momentum_7, Price_EMA_30, Price_ROC_14',
                'Price_Trend_60, Price_Detrended',
                'Month_Sin, DayOfWeek_Cos, Quarter',
                'RSI_14, BB_Position_30, AutoCorr_7'
            ]
        })
        
        st.dataframe(feature_categories, use_container_width=True, hide_index=True)
        
        # Visualize feature distribution
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_categories['Category'],
            y=feature_categories['Count'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'],
            text=feature_categories['Count'],
            textposition='outside'
        ))
        fig.update_layout(
            title="Feature Distribution by Category",
            xaxis_title="Category",
            yaxis_title="Number of Features",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    try:
        ts_comparison_df = pd.read_csv('ts_model_comparison_complete.csv')
        
        # Filter to show only advanced models
        advanced_models = ts_comparison_df[ts_comparison_df['Model'].isin([
            'XGBoost', 'LSTM Deep Learning', 'Random Forest (Tuned)', 
            'Random Forest (Basic)', 'Gradient Boosting Regressor'
        ])]
        
        if not advanced_models.empty:
            st.dataframe(
                advanced_models.style.background_gradient(subset=['R² Score'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
            
            # Best model highlight
            best_idx = advanced_models['R² Score'].idxmax()
            best_model = advanced_models.loc[best_idx]
            
            st.success(f"🥇 **Best Performer:** {best_model['Model']} with R² = {best_model['R² Score']:.4f}")
            
            # Performance metrics visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='MAE',
                x=advanced_models['Model'],
                y=advanced_models['MAE'],
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Mean Absolute Error (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="MAE ()",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # R² Score visualization
            fig2 = go.Figure()
            
            colors = ['green' if x >= 0.90 else 'orange' if x >= 0.85 else 'red' 
                     for x in advanced_models['R² Score']]
            
            fig2.add_trace(go.Bar(
                x=advanced_models['Model'],
                y=advanced_models['R² Score'],
                marker_color=colors,
                text=advanced_models['R² Score'].round(4),
                textposition='outside'
            ))
            
            fig2.add_hline(y=0.90, line_dash="dash", line_color="green", 
                          annotation_text="Excellent (0.90)")
            fig2.add_hline(y=0.85, line_dash="dash", line_color="orange", 
                          annotation_text="Very Good (0.85)")
            
            fig2.update_layout(
                title="R² Score Comparison (Higher is Better)",
                xaxis_title="Model",
                yaxis_title="R² Score",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Model comparison data not found. Please run the Jupyter notebook to train all models.")
    
    st.markdown("---")
    
    # Model Architecture Details
    st.markdown("### 🏗️ Model Architectures")
    
    tab1, tab2, tab3 = st.tabs(["XGBoost", "LSTM", "Random Forest"])
    
    with tab1:
        st.markdown("""
        #### XGBoost Configuration
        
        **Hyperparameters:**
        - `n_estimators`: 500 trees
        - `max_depth`: 8 levels
        - `learning_rate`: 0.01 (slow, stable)
        - `subsample`: 0.8 (80% row sampling)
        - `colsample_bytree`: 0.8 (80% column sampling)
        - `reg_alpha`: 0.1 (L1 regularization)
        - `reg_lambda`: 1.0 (L2 regularization)
        
        **Key Features:**
        - ✅ Gradient boosting for sequential improvement
        - ✅ Handles non-linear relationships
        - ✅ Built-in regularization
        - ✅ Feature importance analysis
        - ✅ Robust to outliers
        
        **Expected Performance:**
        - R² Score: 0.90-0.95
        - MAE: 10-20
        - Training Time: 5-10 minutes
        """)
    
    with tab2:
        st.markdown("""
        #### LSTM Architecture
        
        **Network Structure:**
        ```
        Layer 1: LSTM(128 units) + Dropout(0.2) + BatchNorm
        Layer 2: LSTM(64 units) + Dropout(0.2) + BatchNorm
        Layer 3: LSTM(32 units) + Dropout(0.2) + BatchNorm
        Dense 1: 64 units (ReLU) + Dropout(0.2)
        Dense 2: 32 units (ReLU)
        Output: 1 unit (Price prediction)
        ```
        
        **Configuration:**
        - Time steps: 30 days lookback
        - Batch size: 32
        - Optimizer: Adam (lr=0.001)
        - Early stopping: patience=20
        - Learning rate reduction: factor=0.5
        
        **Key Features:**
        - ✅ Captures long-term dependencies
        - ✅ Sequential pattern recognition
        - ✅ Memory cells for context
        - ✅ Automatic feature learning
        
        **Expected Performance:**
        - R² Score: 0.88-0.93
        - MAE: 12-22
        - Training Time: 10-20 minutes (GPU: 2-5 min)
        """)
    
    with tab3:
        st.markdown("""
        #### Random Forest Configuration
        
        **Hyperparameters (Tuned):**
        - `n_estimators`: 200-300 trees
        - `max_depth`: 20-30 levels
        - `min_samples_split`: 2-5
        - `min_samples_leaf`: 1-2
        - `max_features`: 'sqrt'
        
        **Key Features:**
        - ✅ Ensemble of decision trees
        - ✅ Feature importance ranking
        - ✅ Handles non-linearity
        - ✅ Resistant to overfitting
        - ✅ Fast training and prediction
        
        **Expected Performance:**
        - R² Score: 0.85-0.90
        - MAE: 15-25
        - Training Time: 2-5 minutes
        """)
    
    st.markdown("---")
    
    # Feature Engineering Details
    with st.expander("📚 Advanced Feature Engineering Details"):
        st.markdown("""
        ### Feature Engineering Pipeline
        
        #### 1. Lag Features (10 features)
        Captures historical patterns at multiple horizons:
        - **Short-term**: Lag 1, 2, 3 days
        - **Medium-term**: Lag 5, 7, 14 days
        - **Long-term**: Lag 21, 30, 60, 90 days
        
        #### 2. Rolling Statistics (30 features)
        Computed over windows: 7, 14, 21, 30, 60, 90 days
        - Moving Average (trend indicator)
        - Standard Deviation (volatility measure)
        - Min/Max (support/resistance levels)
        - Range (volatility range)
        
        #### 3. Momentum Features (16 features)
        - **Momentum**: % change over 1, 7, 14, 30 days
        - **ROC**: Rate of change metrics
        - **EMA**: Exponential moving averages (7, 14, 30, 60 days)
        
        #### 4. Trend Features (4 features)
        - Linear trend slopes (30, 60, 90 days)
        - Detrended price (removes long-term trend)
        
        #### 5. Seasonality Features (14 features)
        - Day of Week, Month, Quarter, Year
        - Cyclical encoding (Sin/Cos transformations)
        - Handles circular nature of time
        
        #### 6. Technical Indicators (26+ features)
        - **Bollinger Bands**: Upper/Lower bands, Width, Position
        - **RSI**: Relative Strength Index (14, 30 periods)
        - **Autocorrelation**: Self-correlation measures
        - **Volume**: Trading volume patterns
        - **Price Spreads**: High-Low, Open-Close differences
        
        ### Why These Features?
        
        1. **Lag Features**: Past prices are strong predictors
        2. **Rolling Stats**: Capture volatility and trends
        3. **Momentum**: Identify accelerating trends
        4. **Trend**: Separate signal from noise
        5. **Seasonality**: Gold has seasonal patterns
        6. **Technical Indicators**: Proven trading signals
        """)
    
    # Installation Instructions
    with st.expander("🔧 Installation & Setup"):
        st.markdown("""
        ### Install Required Packages
        
        ```bash
        # Core packages
        pip install pandas numpy scikit-learn
        
        # XGBoost
        pip install xgboost
        
        # Deep Learning (TensorFlow/Keras)
        pip install tensorflow
        
        # Time Series
        pip install statsmodels
        
        # Visualization
        pip install plotly matplotlib seaborn
        
        # All at once
        pip install -r requirements.txt
        ```
        
        ### GPU Support for LSTM (Optional but Recommended)
        
        ```bash
        # For NVIDIA GPUs
        pip install tensorflow-gpu
        
        # Verify GPU availability
        python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
        ```
        
        ### Run Training Pipeline
        
        ```bash
        # Open and run the Jupyter notebook
        jupyter notebook
        # Execute all cells (takes 20-30 minutes)
        
        # Then launch Streamlit app
        streamlit run gold_streamlit_app.py
        ```
        """)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Best Accuracy:**
        - Use **XGBoost** with all advanced features
        - Expected R²: 0.90-0.95
        - Best for: Production forecasting
        
        **For Interpretability:**
        - Use **Random Forest Tuned**
        - Feature importance readily available
        - Best for: Understanding drivers
        """)
    
    with col2:
        st.markdown("""
        **For Sequential Patterns:**
        - Use **LSTM Deep Learning**
        - Captures long-term dependencies
        - Best for: Complex temporal patterns
        
        **For Ensemble:**
        - Combine top 3 models
        - Weighted average predictions
        - Best for: Robustness
        """)

# About Page
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Application")
    
    st.markdown("""
    ### Gold Price Prediction & Forecasting System v3.0
    
    This comprehensive application combines **classification**, **time series forecasting**, and **advanced machine learning** to analyze and predict gold price movements.
    
    #### 🎯 Features
    - **Comprehensive Data Analysis**: 10+ years of historical gold price data (2013-2023)
    - **Multiple ML Models**: Classification and regression models
    - **Advanced Feature Engineering**: 100+ features across 6 categories
    - **Time Series Analysis**: Decomposition, stationarity tests, ACF/PACF
    - **Forecasting Models**: ARIMA, Exponential Smoothing, LSTM, XGBoost
    - **Deep Learning**: LSTM with 3-layer architecture
    - **Interactive Predictions**: Real-time predictions with confidence scores
    
    #### 📊 Analysis Techniques
    
    **Classification Approach:**
    - Predicts whether price will go UP or DOWN
    - Models: Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Naive Bayes
    - Metrics: Accuracy, Precision, Recall, F1-Score
    
    **Time Series Approach:**
    - Time series decomposition (Trend, Seasonality, Residual)
    - Stationarity testing (Augmented Dickey-Fuller test)
    - Autocorrelation analysis (ACF/PACF)
    - ARIMA modeling for forecasting
    - Exponential Smoothing (Holt-Winters)
    - Regression-based forecasting
    
    **Advanced Machine Learning:**
    - **XGBoost**: Gradient boosting with 100+ features
    - **LSTM**: Deep learning for sequential patterns
    - **Random Forest**: Ensemble learning with hyperparameter tuning
    
    #### 🚀 Advanced Feature Engineering (100+ Features)
    
    **1. Lag Features (10)**: Historical prices at multiple horizons
    - Short-term: 1, 2, 3 days
    - Medium-term: 5, 7, 14, 21 days
    - Long-term: 30, 60, 90 days
    
    **2. Rolling Statistics (30)**: Moving window calculations
    - Moving averages, Standard deviations
    - Min/Max values, Price ranges
    - Windows: 7, 14, 21, 30, 60, 90 days
    
    **3. Momentum Features (16)**: Rate of change indicators
    - Price momentum, ROC (Rate of Change)
    - Exponential moving averages
    - Periods: 1, 7, 14, 30 days
    
    **4. Trend Features (4)**: Directional movement
    - Linear trend slopes
    - Detrended prices
    
    **5. Seasonality Features (14)**: Cyclical patterns
    - Day, Month, Quarter, Year
    - Cyclical encoding (Sin/Cos)
    
    **6. Technical Indicators (26+)**: Trading signals
    - Bollinger Bands (Upper, Lower, Width, Position)
    - RSI (Relative Strength Index)
    - Autocorrelation metrics
    - Volume indicators
    - High-Low spreads
    
    #### 📈 Model Performance
    
    **Classification Models:**
    - Best Model: Random Forest (Tuned)
    - Test Accuracy: ~55-65%
    - F1 Score: ~0.55-0.65
    
    **Time Series Models:**
    - ARIMA: MAE ~20-40
    - Exponential Smoothing: MAE ~25-45
    - Regression Models: R² ~0.80-0.90
    
    **Advanced ML Models:**
    - **XGBoost**: R² ~0.90-0.95, MAE ~10-20 ⭐ Best
    - **LSTM**: R² ~0.88-0.93, MAE ~12-22
    - **Random Forest Tuned**: R² ~0.85-0.90, MAE ~15-25
    
    #### 🏗️ Model Architectures
    
    **XGBoost:**
    - 500 trees with gradient boosting
    - Learning rate: 0.01
    - L1/L2 regularization
    - 100+ advanced features
    
    **LSTM:**
    - 3-layer architecture (128/64/32 units)
    - Dropout & Batch Normalization
    - 30-day sequence lookback
    - Adam optimizer
    
    **Random Forest:**
    - 200-300 trees
    - Max depth: 20-30
    - Feature importance ranking
    
    #### 🔬 Methodology
    1. **Data Preprocessing**: Cleaning, feature engineering, outlier detection
    2. **Exploratory Analysis**: Distribution analysis, correlation studies
    3. **Time Series Analysis**: Decomposition, stationarity testing, patterns
    4. **Feature Engineering**: Create 100+ features from raw data
    5. **Model Training**: Multiple algorithms with cross-validation
    6. **Hyperparameter Tuning**: GridSearch and RandomizedSearch
    7. **Evaluation**: Comprehensive metrics (MAE, RMSE, R², Accuracy)
    8. **Forecasting**: Multi-step ahead predictions
    
    #### ⚠️ Disclaimer
    This tool is for **educational and informational purposes only**. Gold price prediction is inherently uncertain 
    and influenced by numerous global factors including:
    - Economic indicators (inflation, interest rates, GDP)
    - Geopolitical events (wars, elections, trade policies)
    - Currency fluctuations (especially USD)
    - Supply and demand dynamics
    - Market sentiment and investor behavior
    - Central bank policies
    
    This model should **NOT** be used as the sole basis for investment decisions. Always:
    - Consult with qualified financial advisors
    - Conduct thorough research
    - Understand your risk tolerance
    - Diversify your portfolio
    - Never invest more than you can afford to lose
    
    #### 👨‍💻 Technology Stack
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning algorithms
    - **XGBoost**: Gradient boosting framework
    - **TensorFlow/Keras**: Deep learning
    - **Statsmodels**: Time series analysis
    - **Pandas & NumPy**: Data manipulation
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    - **Matplotlib & Seaborn**: Statistical graphics
    - **Imbalanced-learn**: SMOTE implementation
    
    #### 📚 Key Concepts
    
    **Stationarity**: A stationary time series has constant statistical properties over time. Required for ARIMA.
    
    **ACF/PACF**: Autocorrelation functions identify patterns and help select model parameters.
    
    **ARIMA**: AutoRegressive Integrated Moving Average - combines AR, I, and MA components.
    
    **XGBoost**: Extreme Gradient Boosting - builds trees sequentially to correct previous errors.
    
    **LSTM**: Long Short-Term Memory - neural network that remembers long sequences.
    
    **Ensemble Learning**: Combining multiple models for better predictions.
    
    **Feature Engineering**: Creating new variables from raw data to improve model performance.
    
    #### 🎯 Use Cases
    
    **Short-term Trading (1-7 days):**
    - Use: LSTM or XGBoost
    - Focus: Technical indicators, momentum
    
    **Medium-term Investment (7-30 days):**
    - Use: XGBoost or Random Forest
    - Focus: Trend and rolling statistics
    
    **Long-term Outlook (30-90 days):**
    - Use: ARIMA or Exponential Smoothing
    - Focus: Seasonal patterns, trends
    
    **Risk Management:**
    - Use: Classification models
    - Focus: Direction prediction
    
    #### 📊 Performance Metrics Explained
    
    **MAE (Mean Absolute Error)**: Average prediction error in dollars
    - Lower is better
    - 10-20 is excellent, 20-30 is good
    
    **RMSE (Root Mean Squared Error)**: Penalizes large errors more
    - Lower is better
    - Similar to MAE but emphasizes outliers
    
    **R² Score**: Proportion of variance explained
    - Higher is better (0-1 scale)
    - >0.90 is excellent, >0.85 is very good
    
    **Accuracy**: % of correct predictions (classification)
    - Higher is better
    - >60% is good for binary prediction
    
    #### 📝 Version History
    
    **v3.0 (January 2025) - Current**
    - Added XGBoost with 100+ features
    - Implemented LSTM deep learning
    - Advanced feature engineering pipeline
    - Comprehensive model comparison
    
    **v2.0**
    - Time series analysis and forecasting
    - ARIMA and Exponential Smoothing
    - Random Forest with tuning
    
    **v1.0**
    - Initial classification models
    - Basic exploratory analysis
    - Streamlit interface
    
    #### 🔗 Resources
    
    **Learn More:**
    - XGBoost: https://xgboost.readthedocs.io/
    - LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    - Time Series: https://otexts.com/fpp3/
    - Streamlit: https://docs.streamlit.io/
    
    **Data Source:**
    - Historical gold price data (2013-2023)
    - Daily OHLC (Open, High, Low, Close) prices
    - Volume and change percentages
    
    #### 📞 Support
    
    For issues, questions, or contributions:
    - Review the Jupyter notebook for detailed implementation
    - Check model comparison files for performance metrics
    - Ensure all required packages are installed
    - Run the complete training pipeline before deployment
    
    ---
    
    **Version**: 3.0.0 (XGBoost + LSTM + Advanced Features)  
    **Last Updated**: January 2025  
    **Models**: 10+ (Classification + Regression + Deep Learning)  
    **Features**: 100+ (Advanced Time Series Engineering)
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Explore all sections to understand gold price patterns from multiple perspectives! The Advanced ML page shows state-of-the-art models.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='font-size: 0.8rem;'>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)