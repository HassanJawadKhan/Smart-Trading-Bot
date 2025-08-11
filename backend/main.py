#!/usr/bin/env python3
"""
FastAPI Stock Prediction Service

This FastAPI application serves the trained Transformer model for stock price predictions.
It loads the exported model and provides RESTful endpoints for real-time predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional, Union
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Stock Prediction API...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    else:
        logger.info("API ready to serve predictions")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Stock Prediction API...")

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="AI-powered stock price prediction using Transformer neural networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
scalers = None
model_metadata = None
feature_columns = None
device = None

# Pydantic models for request/response
class StockSymbol(BaseModel):
    symbol: str
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        v = v.upper().strip()
        if not v or len(v) > 10:
            raise ValueError("Symbol must be 1-10 characters")
        return v

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: Optional[int] = 1
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        return v.upper().strip()
    
    @field_validator('days_ahead')
    @classmethod
    def validate_days_ahead(cls, v):
        if v < 1 or v > 30:
            raise ValueError("days_ahead must be between 1 and 30")
        return v

class BatchPredictionRequest(BaseModel):
    symbols: List[str]
    days_ahead: Optional[int] = 1
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        if not v or len(v) > 50:
            raise ValueError("Must provide 1-50 symbols")
        return [s.upper().strip() for s in v]

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    predicted_change_percent: float
    confidence_score: float
    days_ahead: int
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supported_symbols: List[str]
    model_info: Dict
    timestamp: str

# Technical Indicators (same as training script)
class TechnicalIndicators:
    """Technical analysis indicators for feature engineering"""
    
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band, sma
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high, low, close, window=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def obv(close, volume):
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()

# Positional Encoding (same as training script)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Stock Transformer Model (same as training script)
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, 
                 dim_feedforward=512, dropout=0.1, max_len=5000):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        
        transformer_out = self.transformer_encoder(src)
        
        last_hidden = transformer_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        output = self.prediction_head(last_hidden)
        
        return output

class StockDataProcessor:
    """Process stock data for prediction"""
    
    def __init__(self, feature_columns, scalers, sequence_length=60):
        self.feature_columns = feature_columns
        self.scalers = scalers
        self.sequence_length = sequence_length
    
    def fetch_recent_data(self, symbol, days_back=90):
        """Fetch recent stock data for prediction"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            
            return data
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}: {str(e)}")
    
    def engineer_features(self, data):
        """Engineer the same features used in training"""
        df = data.copy()
        
        # Basic price features
        df['price_change'] = df['Close'].pct_change()
        df['price_change_2d'] = df['Close'].pct_change(periods=2)
        df['price_change_5d'] = df['Close'].pct_change(periods=5)
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['volume_change'] = df['Volume'].pct_change()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = TechnicalIndicators.sma(df['Close'], window)
            df[f'ema_{window}'] = TechnicalIndicators.ema(df['Close'], window)
            df[f'close_sma_{window}_ratio'] = df['Close'] / df[f'sma_{window}']
            df[f'volume_sma_{window}'] = TechnicalIndicators.sma(df['Volume'], window)
        
        # Technical indicators
        df['rsi'] = TechnicalIndicators.rsi(df['Close'])
        df['rsi_14'] = TechnicalIndicators.rsi(df['Close'], 14)
        df['rsi_30'] = TechnicalIndicators.rsi(df['Close'], 30)
        
        # MACD
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_sma = TechnicalIndicators.bollinger_bands(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_upper - bb_lower
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(
            df['High'], df['Low'], df['Close']
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # OBV
        df['obv'] = TechnicalIndicators.obv(df['Close'], df['Volume'])
        df['obv_sma'] = TechnicalIndicators.sma(df['obv'], 10)
        
        # Volatility features
        df['price_volatility_5d'] = df['Close'].rolling(5).std()
        df['price_volatility_10d'] = df['Close'].rolling(10).std()
        df['price_volatility_20d'] = df['Close'].rolling(20).std()
        df['volume_volatility_5d'] = df['Volume'].rolling(5).std()
        
        # Momentum features
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Time features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        
        return df
    
    def prepare_prediction_data(self, symbol):
        """Prepare data for model prediction"""
        # Fetch recent data
        raw_data = self.fetch_recent_data(symbol)
        
        # Engineer features
        processed_data = self.engineer_features(raw_data)
        
        # Remove NaN values
        processed_data = processed_data.dropna()
        
        if len(processed_data) < self.sequence_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for {symbol}. Need at least {self.sequence_length} days"
            )
        
        # Select feature columns that match training
        available_features = []
        for col in self.feature_columns:
            if col in processed_data.columns:
                available_features.append(col)
        
        if len(available_features) != len(self.feature_columns):
            logger.warning(f"Feature mismatch for {symbol}. Expected {len(self.feature_columns)}, got {len(available_features)}")
        
        # Get features
        X = processed_data[available_features].values
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        # Get last sequence for prediction
        X_seq = X_scaled[-self.sequence_length:]
        
        # Current price for reference
        current_price = float(processed_data['Close'].iloc[-1])
        
        return X_seq, current_price

def load_model():
    """Load the trained model and associated files"""
    global model, scalers, model_metadata, feature_columns, device
    
    try:
        # Define paths relative to training directory
        training_dir = Path("../training")
        model_path = training_dir / "stock_transformer_model.pth"
        scalers_path = training_dir / "model_scalers.pkl"
        metadata_path = training_dir / "model_metadata.json"
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers file not found: {scalers_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config']
        feature_columns = checkpoint['feature_columns']
        
        # Initialize model
        model = StockTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load scalers
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model supports {len(feature_columns)} features")
        logger.info(f"Trained on symbols: {model_metadata.get('symbols', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def calculate_confidence(prediction_value):
    """Calculate a simple confidence score based on prediction magnitude"""
    # Simple heuristic - closer to 0% change = higher confidence
    # This can be improved with ensemble methods or prediction intervals
    abs_change = abs(prediction_value)
    if abs_change < 0.01:  # < 1% change
        return 0.9
    elif abs_change < 0.03:  # < 3% change
        return 0.8
    elif abs_change < 0.05:  # < 5% change
        return 0.7
    elif abs_change < 0.10:  # < 10% change
        return 0.6
    else:
        return 0.5

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Stock Price Prediction API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        supported_symbols=model_metadata.get('symbols', []) if model_metadata else [],
        model_info={
            "features": len(feature_columns) if feature_columns else 0,
            "training_date": model_metadata.get('training_date', 'Unknown') if model_metadata else 'Unknown',
            "device": str(device) if device else 'Unknown'
        },
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """Predict stock price for a single symbol"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Initialize data processor
        processor = StockDataProcessor(feature_columns, scalers, model_metadata['sequence_length'])
        
        # Prepare data
        X_seq, current_price = processor.prepare_prediction_data(request.symbol)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(X_tensor)
            prediction_scaled = prediction.cpu().numpy()[0, 0]
        
        # Inverse transform prediction
        prediction_change = scalers['target'].inverse_transform([[prediction_scaled]])[0, 0]
        
        # Calculate predicted price
        predicted_price = current_price * (1 + prediction_change)
        predicted_change_percent = prediction_change * 100
        
        # Calculate confidence
        confidence = calculate_confidence(prediction_change)
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=round(current_price, 2),
            predicted_price=round(predicted_price, 2),
            predicted_change_percent=round(predicted_change_percent, 2),
            confidence_score=round(confidence, 2),
            days_ahead=request.days_ahead,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict_stocks(request: BatchPredictionRequest):
    """Predict stock prices for multiple symbols"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    errors = []
    
    for symbol in request.symbols:
        try:
            prediction_request = PredictionRequest(symbol=symbol, days_ahead=request.days_ahead)
            result = await predict_stock(prediction_request)
            results.append(result)
        except Exception as e:
            errors.append(f"{symbol}: {str(e)}")
            logger.warning(f"Failed to predict {symbol}: {e}")
    
    if not results and errors:
        raise HTTPException(status_code=400, detail=f"All predictions failed. Errors: {errors}")
    
    return results

@app.get("/supported-symbols", response_model=List[str])
async def get_supported_symbols():
    """Get list of symbols the model was trained on"""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    
    return model_metadata.get('symbols', [])

@app.post("/reload-model")
async def reload_model():
    """Reload the model (useful for updates)"""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
