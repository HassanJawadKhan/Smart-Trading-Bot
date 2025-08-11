#!/usr/bin/env python3
"""
Stock Price Prediction Transformer Model Training Script

This script implements a comprehensive Transformer-based model for stock price prediction
with technical analysis feature engineering, proper data preprocessing, and model export.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
import logging
import pickle
import json
from datetime import datetime, timedelta
import math
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical analysis indicators for feature engineering"""
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band, sma
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()

class StockDataProcessor:
    """Data processor for stock data with feature engineering"""
    
    def __init__(self, symbols, start_date, end_date, sequence_length=60):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.scalers = {}
        self.feature_columns = []
        
    def fetch_data(self):
        """Fetch stock data for multiple symbols"""
        logger.info(f"Fetching data for symbols: {self.symbols}")
        all_data = []
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                data['Symbol'] = symbol
                all_data.append(data)
                logger.info(f"Downloaded {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be fetched for any symbol")
        
        combined_data = pd.concat(all_data, ignore_index=False)
        logger.info(f"Total combined data points: {len(combined_data)}")
        return combined_data
    
    def engineer_features(self, data):
        """Engineer technical analysis features"""
        logger.info("Engineering technical features...")
        
        # Create a copy to avoid modifying original data
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
        
        # ATR (volatility)
        df['atr'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # On-Balance Volume
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
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        
        # Target variable (next day's closing price change)
        df['target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        logger.info(f"Engineered {len(df.columns)} features")
        return df
    
    def prepare_data(self):
        """Prepare data for training"""
        # Fetch raw data
        raw_data = self.fetch_data()
        
        # Engineer features
        processed_data = self.engineer_features(raw_data)
        
        # Remove rows with NaN values
        processed_data = processed_data.dropna()
        
        # Select feature columns (exclude target and non-numeric columns)
        exclude_cols = ['Symbol', 'target']
        feature_cols = [col for col in processed_data.columns 
                       if col not in exclude_cols and 
                       processed_data[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_cols
        logger.info(f"Selected {len(feature_cols)} features for training")
        
        # Prepare features and targets
        X = processed_data[feature_cols].values
        y = processed_data['target'].values
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Scale targets
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        self.scalers['target'] = target_scaler
        
        logger.info(f"Data shape: X={X_scaled.shape}, y={y_scaled.shape}")
        return X_scaled, y_scaled, processed_data.index

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences"""
    
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_val = self.y[idx + self.sequence_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
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

class StockTransformer(nn.Module):
    """Transformer model for stock price prediction"""
    
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
        
        # Multi-layer prediction head with residual connections
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
        # Project input to model dimension
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = src.transpose(0, 1)  # (seq_len, batch, d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(src)
        
        # Use the last time step for prediction
        last_hidden = transformer_out[:, -1, :]  # (batch, d_model)
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Generate prediction
        output = self.prediction_head(last_hidden)
        
        return output

class StockPredictor:
    """Main class for training stock prediction model"""
    
    def __init__(self, symbols, sequence_length=60, test_size=0.2, random_state=42):
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data_processor = None
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, start_date, end_date):
        """Prepare training and validation data"""
        logger.info("Preparing data...")
        
        # Initialize data processor
        self.data_processor = StockDataProcessor(
            self.symbols, start_date, end_date, self.sequence_length
        )
        
        # Prepare data
        X, y, dates = self.data_processor.prepare_data()
        
        # Split data temporally (not randomly) to avoid data leakage
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = StockDataset(X_train, y_train, self.sequence_length)
        val_dataset = StockDataset(X_val, y_val, self.sequence_length)
        
        logger.info(f"Training sequences: {len(train_dataset)}")
        logger.info(f"Validation sequences: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def create_model(self, input_dim):
        """Create the transformer model"""
        model = StockTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=6,
            dim_feedforward=512,
            dropout=0.1
        )
        
        return model.to(self.device)
    
    def train_model(self, train_dataset, val_dataset, epochs=100, batch_size=64, 
                   learning_rate=0.001, patience=15):
        """Train the transformer model"""
        logger.info("Starting model training...")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        # Initialize model
        input_dim = len(self.data_processor.feature_columns)
        self.model = self.create_model(input_dim)
        
        # Loss function and optimizer
        criterion = nn.HuberLoss(delta=0.1)  # More robust to outliers than MSE
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_stock_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_stock_model.pth'))
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return train_losses, val_losses
    
    def save_model(self, model_path='stock_transformer_model.pth'):
        """Save the trained model and associated metadata"""
        logger.info(f"Saving model to {model_path}")
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': len(self.data_processor.feature_columns),
                'd_model': 128,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 512,
                'dropout': 0.1
            },
            'feature_columns': self.data_processor.feature_columns,
            'sequence_length': self.sequence_length,
            'symbols': self.symbols
        }, model_path)
        
        # Save scalers
        with open('model_scalers.pkl', 'wb') as f:
            pickle.dump(self.data_processor.scalers, f)
        
        # Save metadata
        metadata = {
            'symbols': self.symbols,
            'sequence_length': self.sequence_length,
            'feature_columns': self.data_processor.feature_columns,
            'training_date': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model, scalers, and metadata saved successfully")

def main():
    """Main training function"""
    # Configuration - Pakistani Stocks and Gold
    SYMBOLS = [
        # Gold ETFs and commodities
        'GLD',    # SPDR Gold Shares ETF
        'GOLD',   # Barrick Gold Corporation
        'IAU',    # iShares Gold Trust ETF
        'GDX',    # VanEck Gold Miners ETF
        'GDXJ',   # VanEck Junior Gold Miners ETF
        
        # Pakistani companies trading on international exchanges
        # Note: Most PSX stocks are not directly available on Yahoo Finance
        # Using Pakistani companies with international presence or ADRs
        'SCCO',   # Southern Copper Corporation (mining - relevant to Pakistani mining sector)
        'FCX',    # Freeport-McMoRan (copper/gold mining)
        
        # Pakistani-focused ETFs and international proxies
        'EPOL',   # iShares MSCI Poland ETF (emerging market proxy)
        'EEM',    # iShares MSCI Emerging Markets ETF
        'VWO',    # Vanguard Emerging Markets Stock ETF
        'IEMG',   # iShares Core MSCI Emerging Markets ETF
        
        # Regional banks and financials (similar to Pakistani banking sector)
        'HDB',    # HDFC Bank Limited (Indian bank - regional proxy)
        'IBN',    # ICICI Bank Limited (Indian bank - regional proxy)
        'WF',     # Woori Financial Group (Korean bank - emerging market banking)
        
        # Textile and manufacturing (key Pakistani industries)
        'TPG',    # TPG Inc (industrial proxy)
        'LPX',    # Louisiana-Pacific Corporation (materials)
        
        # Energy sector (important for Pakistan)
        'PTR',    # PetroChina Company Limited
        'CEO',    # CNOOC Limited
        'SU',     # Suncor Energy Inc
        'E',      # Eni S.p.A
        
        # Telecommunications (key Pakistani sector)
        'VZ',     # Verizon Communications Inc
        'T',      # AT&T Inc
        'TMUS',   # T-Mobile US Inc
        
        # Fertilizer and agriculture (major Pakistani industry)
        'CF',     # CF Industries Holdings
        'MOS',    # The Mosaic Company
        'NTR',    # Nutrien Ltd
        'FMC',    # FMC Corporation
        
        # Cement and construction materials (key Pakistani sector)
        'VMC',    # Vulcan Materials Company
        'MLM',    # Martin Marietta Materials
    ]  # Pakistani sector-focused and regional proxy stocks with gold
    
    # Use last 5 years of data
    START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    SEQUENCE_LENGTH = 60  # 60 days of historical data
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    
    logger.info("=" * 60)
    logger.info("STOCK PRICE PREDICTION TRANSFORMER TRAINING")
    logger.info("=" * 60)
    logger.info(f"Symbols: {len(SYMBOLS)} stocks")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH} days")
    logger.info(f"Training epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    
    try:
        # Initialize predictor
        predictor = StockPredictor(
            symbols=SYMBOLS,
            sequence_length=SEQUENCE_LENGTH,
            test_size=0.2,
            random_state=42
        )
        
        # Prepare data
        train_dataset, val_dataset = predictor.prepare_data(START_DATE, END_DATE)
        
        # Train model
        train_losses, val_losses = predictor.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            patience=20
        )
        
        # Save model
        predictor.save_model()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("Files generated:")
        logger.info("- stock_transformer_model.pth (model weights)")
        logger.info("- model_scalers.pkl (feature scalers)")
        logger.info("- model_metadata.json (model metadata)")
        logger.info("- best_stock_model.pth (best checkpoint)")
        logger.info("=" * 60)
        
        # Final statistics
        logger.info(f"Final training loss: {train_losses[-1]:.6f}")
        logger.info(f"Final validation loss: {val_losses[-1]:.6f}")
        logger.info(f"Total features engineered: {len(predictor.data_processor.feature_columns)}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
