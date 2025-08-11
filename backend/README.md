# Stock Prediction API

A high-performance FastAPI backend for AI-powered stock price prediction using Transformer neural networks.

## Features

- 🤖 **AI-Powered Predictions**: Uses trained Transformer model with 50+ technical indicators
- ⚡ **Fast & Scalable**: Built with FastAPI for high performance
- 🔄 **Real-time Data**: Fetches live stock data via yfinance
- 📊 **Comprehensive Analysis**: Technical indicators, volatility measures, momentum features
- 🎯 **Multiple Endpoints**: Single predictions, batch processing, health checks
- 📖 **Auto Documentation**: Interactive API docs with Swagger/ReDoc
- 🌐 **CORS Ready**: Configured for frontend integration

## API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check and model status  
- `POST /predict` - Single stock prediction
- `POST /batch-predict` - Batch predictions for multiple stocks
- `GET /supported-symbols` - List of supported stock symbols
- `POST /reload-model` - Reload the AI model

### Documentation

- `GET /docs` - Interactive Swagger documentation
- `GET /redoc` - ReDoc documentation

## Quick Start

### Prerequisites

Make sure the trained model files are in the `../training/` directory:
- `stock_transformer_model.pth` - Model weights
- `model_scalers.pkl` - Feature scalers  
- `model_metadata.json` - Model metadata

### Installation & Startup

```bash
# Install dependencies with UV
uv sync

# Start the API server
uv run python start.py

# Or run directly
uv run python main.py
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Alternative Startup

```bash
# Using uvicorn directly
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage Examples

### Single Stock Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 1}'
```

Response:
```json
{
  "symbol": "AAPL",
  "current_price": 150.25,
  "predicted_price": 152.10,
  "predicted_change_percent": 1.23,
  "confidence_score": 0.85,
  "days_ahead": 1,
  "timestamp": "2024-01-15T10:30:00.123456",
  "model_version": "1.0.0"
}
```

### Batch Predictions

```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT", "GOOGL"], "days_ahead": 1}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Testing

Run the test script to verify all endpoints:

```bash
uv run python test_api.py
```

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)  
- `LOG_LEVEL`: Logging level (default: info)

### Model Configuration

The API automatically loads model configuration from:
- Model checkpoint file
- Scalers for feature normalization
- Metadata with training information

## Production Deployment

For production deployment:

1. **Set specific CORS origins** in `main.py`:
```python
allow_origins=["https://your-frontend-domain.com"]
```

2. **Use production WSGI server**:
```bash
uv add gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

3. **Configure environment variables**:
```bash
export PORT=8000
export LOG_LEVEL=warning
```

4. **Add authentication/rate limiting** as needed

## Architecture

### Model Pipeline
1. **Data Fetching**: Real-time stock data via yfinance
2. **Feature Engineering**: 50+ technical indicators
3. **Preprocessing**: Same scaling used in training
4. **Prediction**: Transformer neural network inference
5. **Post-processing**: Price calculation and confidence scoring

### Key Components
- **FastAPI Application**: High-performance async web framework
- **Transformer Model**: 6-layer neural network with attention
- **Data Processor**: Technical analysis and feature engineering
- **CORS Middleware**: Cross-origin request support

## Error Handling

The API includes comprehensive error handling:
- **404**: Symbol not found or no data available
- **400**: Invalid input parameters or insufficient data
- **500**: Internal server errors with detailed messages
- **503**: Model not loaded or unavailable

## Monitoring

Check API health and model status:
- **Health endpoint**: `/health`
- **Model metrics**: Features count, training date, device info
- **Supported symbols**: Full list of trainable stocks

## Limitations

- Predictions are for informational purposes only
- Model accuracy depends on market conditions
- Requires internet connection for real-time data
- Limited to stocks with sufficient historical data
