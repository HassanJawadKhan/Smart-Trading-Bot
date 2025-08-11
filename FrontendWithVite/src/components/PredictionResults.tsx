import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Target, Clock, AlertTriangle, Trash2 } from 'lucide-react';
import { PredictionResponse, PredictionStatus } from '../types';
import { formatCurrency, formatPercentage, formatConfidence, formatRelativeTime, getPredictionColor, getConfidenceColor } from '../utils';

interface PredictionResultsProps {
  predictions: PredictionResponse[];
  status: PredictionStatus;
  error: string | null;
  onClearPredictions: () => void;
  onClearError: () => void;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({
  predictions,
  status,
  error,
  onClearPredictions,
  onClearError,
}) => {
  if (error) {
    return (
      <div className="prediction-card border-l-4 border-l-red-500 bg-red-50/30">
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-6 h-6 text-red-500 mt-1" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-red-900 mb-2">
              Prediction Failed
            </h3>
            <p className="text-red-700 mb-4">{error}</p>
            <button
              onClick={onClearError}
              className="btn-secondary text-sm"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (predictions.length === 0 && status === 'idle') {
    return (
      <div className="prediction-card text-center py-12">
        <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <TrendingUp className="w-8 h-8 text-white" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Ready for Predictions
        </h3>
        <p className="text-gray-600 max-w-md mx-auto">
          Enter a stock symbol to get AI-powered price predictions with confidence scores and technical analysis.
        </p>
      </div>
    );
  }

  if (status === 'loading' && predictions.length === 0) {
    return (
      <div className="prediction-card text-center py-12">
        <div className="loading-spinner mx-auto mb-4"></div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Generating Predictions
        </h3>
        <p className="text-gray-600">
          Our AI model is analyzing market data and technical indicators...
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold gradient-text">
            Prediction Results
          </h2>
          <p className="text-gray-600 text-sm">
            {predictions.length} prediction{predictions.length !== 1 ? 's' : ''} generated
          </p>
        </div>
        {predictions.length > 0 && (
          <button
            onClick={onClearPredictions}
            className="btn-secondary flex items-center space-x-2 text-sm"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear All</span>
          </button>
        )}
      </div>

      {/* Predictions List */}
      <div className="space-y-4">
        {predictions.map((prediction) => {
          const isPositive = prediction.predicted_change_percent > 0;
          const isNeutral = Math.abs(prediction.predicted_change_percent) < 0.1;
          
          return (
            <div
              key={`${prediction.symbol}-${prediction.timestamp}`}
              className={`prediction-card border-l-4 ${
                isPositive ? 'border-l-green-500 bg-green-50/30' : 
                isNeutral ? 'border-l-gray-500 bg-gray-50/30' : 
                'border-l-red-500 bg-red-50/30'
              }`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`p-3 rounded-xl ${isPositive ? 'bg-green-100' : isNeutral ? 'bg-gray-100' : 'bg-red-100'}`}>
                    {isPositive ? (
                      <TrendingUp className="w-6 h-6 text-green-600" />
                    ) : isNeutral ? (
                      <DollarSign className="w-6 h-6 text-gray-600" />
                    ) : (
                      <TrendingDown className="w-6 h-6 text-red-600" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">
                      {prediction.symbol}
                    </h3>
                    <p className="text-sm text-gray-600 flex items-center space-x-2">
                      <Clock className="w-3 h-3" />
                      <span>{formatRelativeTime(prediction.timestamp)}</span>
                    </p>
                  </div>
                </div>
                
                <div className={`px-3 py-1 rounded-full border text-sm font-medium ${getConfidenceColor(prediction.confidence_score)}`}>
                  {formatConfidence(prediction.confidence_score)}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="metric-card">
                  <div className="flex items-center space-x-2">
                    <DollarSign className="w-4 h-4 text-blue-500" />
                    <div>
                      <p className="text-xs font-medium text-gray-600 uppercase tracking-wider">Current Price</p>
                      <p className="text-lg font-bold text-gray-900">
                        {formatCurrency(prediction.current_price)}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="flex items-center space-x-2">
                    <Target className="w-4 h-4 text-purple-500" />
                    <div>
                      <p className="text-xs font-medium text-gray-600 uppercase tracking-wider">Predicted Price</p>
                      <p className="text-lg font-bold text-gray-900">
                        {formatCurrency(prediction.predicted_price)}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="flex items-center space-x-2">
                    {isPositive ? (
                      <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500" />
                    )}
                    <div>
                      <p className="text-xs font-medium text-gray-600 uppercase tracking-wider">Expected Change</p>
                      <p className={`text-lg font-bold ${getPredictionColor(prediction.predicted_change_percent)}`}>
                        {formatPercentage(prediction.predicted_change_percent)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm text-gray-500 pt-4 border-t border-gray-200">
                <span>Model: v{prediction.model_version}</span>
                <span>Timeframe: {prediction.days_ahead} day{prediction.days_ahead !== 1 ? 's' : ''}</span>
                <span>Confidence: {formatConfidence(prediction.confidence_score)}</span>
              </div>
            </div>
          );
        })}
      </div>

      {status === 'loading' && predictions.length > 0 && (
        <div className="prediction-card text-center py-8 border-dashed border-2 border-gray-300">
          <div className="loading-spinner mx-auto mb-3"></div>
          <p className="text-gray-600">Loading additional predictions...</p>
        </div>
      )}
    </div>
  );
};

export default PredictionResults;
