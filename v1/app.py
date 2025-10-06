from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import logging
import os
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bitcoin Price Prediction API",
    description="AI-powered Bitcoin price prediction using LSTM neural networks",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None

# Load model and scaler with error handling
def load_models():
    global model, scaler
    try:
        # Try to load TensorFlow model
        try:
            from tensorflow.keras.models import load_model
            if os.path.exists("./models/bitcoin_lstm_model.h5"):
                model = load_model("./models/bitcoin_lstm_model.h5")
                logger.info("✅ LSTM model loaded successfully")
            else:
                logger.warning("⚠️ LSTM model file not found, will use fallback")
        except ImportError:
            logger.warning("⚠️ TensorFlow not available, will use fallback model")
        
        # Load scaler
        if os.path.exists("./models/bitcoin_scaler.save"):
            scaler = joblib.load("./models/bitcoin_scaler.save")
            logger.info("✅ Scaler loaded successfully")
        else:
            logger.warning("⚠️ Scaler file not found, will create new one")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

class PredictionRequest(BaseModel):
    days: int = Field(default=1, ge=1, le=30, description="Number of days to predict (1-30)")
    model_type: Optional[str] = Field(default="auto", description="Model type: lstm, linear, or auto")

class PredictionResponse(BaseModel):
    date: str
    predicted_close: float
    confidence: Optional[float] = None
    trend: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    predictions: List[PredictionResponse]
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    current_btc_price: Optional[float] = None
    last_updated: str

def fetch_data(days=90, interval="1d"):
    """Fetch Bitcoin data with error handling"""
    try:
        logger.info(f"📊 Fetching {days} days of Bitcoin data...")
        if days < 30:
            days = 30
        df = yf.download("BTC-USD", period=f"{days}d", interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data received from Yahoo Finance")
        df = df[['Close', 'Volume', 'High', 'Low', 'Open']].dropna()
        if len(df) < 30:
            raise ValueError(f"Insufficient data: only {len(df)} rows received")
        logger.info(f"✅ Successfully fetched {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"❌ Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

def create_features(df):
    """Create technical indicators"""
    try:
        data = df.copy()
        data['MA7'] = data['Close'].rolling(7, min_periods=1).mean()
        data['MA21'] = data['Close'].rolling(21, min_periods=1).mean()
        data['MA50'] = data['Close'].rolling(50, min_periods=1).mean()
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        rolling_mean = data['Close'].rolling(20, min_periods=1).mean()
        rolling_std = data['Close'].rolling(20, min_periods=1).std()
        data['BB_upper'] = rolling_mean + 2 * rolling_std
        data['BB_lower'] = rolling_mean - 2 * rolling_std
        data['Volatility'] = data['Close'].pct_change().rolling(14, min_periods=1).std()
        volume_ma = data['Volume'].rolling(14, min_periods=1).mean()
        data['Volume_MA'] = volume_ma
        data['Volume_ratio'] = data['Volume'] / (volume_ma + 1e-10)
        data['Momentum'] = data['Close'] - data['Close'].shift(4)
        data['ROC'] = data['Close'].pct_change(periods=4) * 100
        data = data.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').dropna()
        return data
    except Exception as e:
        logger.error(f"❌ Error creating features: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create features: {e}")

def predict_with_lstm(df, days):
    """Make predictions using LSTM"""
    try:
        if model is None or scaler is None:
            raise ValueError("LSTM model or scaler not available")
        df_feat = create_features(df)
        features = ['Close','Volume','High','Low','Open','MA7','MA21','RSI','MACD','Volatility']
        avail = [f for f in features if f in df_feat.columns]
        X = df_feat[avail].values
        # Determine scaler feature count
        try:
            n_feat = scaler.n_features_in_
        except AttributeError:
            n_feat = scaler.mean_.shape[0]
        # Adjust dimensions
        if X.shape[1] != n_feat:
            if X.shape[1] < n_feat:
                pad = np.zeros((X.shape[0], n_feat - X.shape[1]))
                X = np.hstack([X, pad])
            else:
                X = X[:, :n_feat]
        Xs = scaler.transform(X)
        seq_len = min(60, len(Xs)-1)
        seq = Xs[-seq_len:].copy()
        preds = []
        for _ in range(days):
            p = float(model.predict(seq.reshape(1, seq_len, n_feat), verbose=0)[0][0])
            preds.append(p)
            row = seq[-1].copy()
            row[0] = p
            seq = np.vstack([seq[1:], row])
        dum = np.zeros((len(preds), n_feat)); dum[:,0]=preds
        return scaler.inverse_transform(dum)[:,0].tolist()
    except Exception as e:
        logger.error(f"❌ LSTM prediction failed: {e}")
        return None

def predict_with_fallback(df, days):
    """Make predictions using fallback linear model"""
    try:
        df_feat = create_features(df)
        key = ['Close','RSI','MACD','Momentum']
        cols = [c for c in key if c in df_feat.columns]
        X = df_feat[cols].values
        # load or train fallback
        try:
            m = joblib.load("./models/fallback_model.save")
            poly = joblib.load("./models/poly_features.save")
        except:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            Xp = poly.fit_transform(X)
            m = LinearRegression(); m.fit(Xp[:-1], df_feat['Close'].values[1:])
            joblib.dump(m,"./models/fallback_model.save"); joblib.dump(poly,"./models/poly_features.save")
        preds=[]
        lv=X[-1].copy()
        for _ in range(days):
            xp=poly.transform(lv.reshape(1,-1))
            p=float(m.predict(xp)[0]); preds.append(p)
            lv[0]=p
        return preds
    except Exception as e:
        logger.error(f"❌ Fallback prediction failed: {e}")
        return None

def calculate_confidence_and_trend(preds, curr):
    conf=[]; trends=[]
    for i,p in enumerate(preds):
        pc=abs(p-curr)/curr; c=max(0.3,min(0.95,1-pc*2)); conf.append(c)
        if i==0: t="up" if p>curr else "down" if p<curr else "sideways"
        else: t="up" if p>preds[i-1] else "down" if p<preds[i-1] else "sideways"
        trends.append(t)
    return conf, trends

@app.get("/",response_model=dict)
def home():
    return {"message":"API running","endpoints":["/predict","/health","/current-price","/historical/{days}"]}

@app.get("/health",response_model=HealthResponse)
def health():
    try:
        cb=None
        try:
            d=yf.download("BTC-USD",period="1d",interval="1d",progress=False)
            if not d.empty: cb=float(d['Close'].iloc[-1])
        except: pass
        return HealthResponse(status="healthy",
            models_loaded={"lstm":model is not None,"scaler":scaler is not None,"fallback":True},
            current_btc_price=cb,last_updated=datetime.now().isoformat())
    except Exception as e:
        return HealthResponse(status="degraded",models_loaded={"error":str(e)},last_updated=datetime.now().isoformat())

@app.post("/predict",response_model=APIResponse)
def predict(data:PredictionRequest):
    try:
        df=fetch_data(90); curr=float(df['Close'].iloc[-1])
        preds=None; method="none"
        if data.model_type in ["lstm","auto"]:
            preds=predict_with_lstm(df,data.days)
            if preds: method="lstm"
        if not preds and data.model_type in ["linear","auto"]:
            preds=predict_with_fallback(df,data.days)
            if preds: method="linear"
        if not preds:
            rt=(df['Close'].iloc[-1]-df['Close'].iloc[-7])/7
            preds=[curr+rt*(i+1) for i in range(data.days)]; method="trend"
        if not isinstance(preds,list) or any((not isinstance(p,(int,float))) for p in preds):
            raise ValueError("Invalid predictions")
        conf, trends=calculate_confidence_and_trend(preds,curr)
        last=df.index[-1]; fdates=[last+timedelta(days=i+1) for i in range(data.days)]
        resp=[PredictionResponse(date=d.strftime("%Y-%m-%d"),
               predicted_close=round(float(p),2),
               confidence=round(conf[i],3),trend=trends[i]) 
               for i,(d,p) in enumerate(zip(fdates,preds))]
        return APIResponse(success=True,message=f"Predicted {data.days} days",
            predictions=resp,metadata={"method":method,"current_price":round(curr,2),"generated_at":datetime.now().isoformat()})
    except HTTPException: raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/current-price")
def current_price():
    try:
        df=fetch_data(1); cp=float(df['Close'].iloc[-1])
        return {"current_price":round(cp,2),"timestamp":datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/historical/{days}")
def historical(days:int=30):
    try:
        days=min(days,365)
        df=fetch_data(days); ff=create_features(df)
        sample=ff.tail(min(100,len(ff)))
        return {"data":sample.to_dict('records'),"total":len(ff),"returned":len(sample),"period":f"{days}d"}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000,reload=True)
