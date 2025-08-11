import React, { useState } from 'react';
import { Search, Zap, Plus, X } from 'lucide-react';
import toast from 'react-hot-toast';
import { PredictionRequest, BatchPredictionRequest } from '../types';
import { isValidStockSymbol } from '../utils';

interface PredictionFormProps {
  onSinglePrediction: (request: PredictionRequest) => Promise<any>;
  onBatchPrediction: (request: BatchPredictionRequest) => Promise<any>;
  isLoading: boolean;
  supportedSymbols: string[];
  isApiHealthy: boolean;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onSinglePrediction,
  onBatchPrediction,
  isLoading,
  supportedSymbols,
  isApiHealthy,
}) => {
  const [symbol, setSymbol] = useState('');
  const [batchSymbols, setBatchSymbols] = useState<string[]>([]);
  const [newBatchSymbol, setNewBatchSymbol] = useState('');
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  const handleSingleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!symbol.trim()) {
      toast.error('Please enter a stock symbol');
      return;
    }

    if (!isValidStockSymbol(symbol)) {
      toast.error('Please enter a valid stock symbol (1-5 letters)');
      return;
    }

    if (!isApiHealthy) {
      toast.error('API is not available. Please check connection.');
      return;
    }

    const result = await onSinglePrediction({ symbol: symbol.toUpperCase() });
    
    if (result) {
      toast.success(`Prediction generated for ${symbol.toUpperCase()}`);
      setSymbol('');
    }
  };

  const handleBatchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (batchSymbols.length === 0) {
      toast.error('Please add at least one stock symbol');
      return;
    }

    if (!isApiHealthy) {
      toast.error('API is not available. Please check connection.');
      return;
    }

    const result = await onBatchPrediction({ symbols: batchSymbols });
    
    if (result) {
      toast.success(`Batch prediction generated for ${batchSymbols.length} symbols`);
      setBatchSymbols([]);
    }
  };

  const addBatchSymbol = () => {
    const trimmedSymbol = newBatchSymbol.trim().toUpperCase();
    
    if (!trimmedSymbol) {
      toast.error('Please enter a symbol');
      return;
    }

    if (!isValidStockSymbol(trimmedSymbol)) {
      toast.error('Please enter a valid stock symbol');
      return;
    }

    if (batchSymbols.includes(trimmedSymbol)) {
      toast.error('Symbol already added');
      return;
    }

    if (batchSymbols.length >= 10) {
      toast.error('Maximum 10 symbols allowed');
      return;
    }

    setBatchSymbols([...batchSymbols, trimmedSymbol]);
    setNewBatchSymbol('');
  };

  const removeBatchSymbol = (symbolToRemove: string) => {
    setBatchSymbols(batchSymbols.filter(s => s !== symbolToRemove));
  };

  const popularSymbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META'];

  return (
    <div className="prediction-card">
      <div className="mb-6">
        <h2 className="text-xl font-bold gradient-text mb-2">
          AI Stock Prediction
        </h2>
        <p className="text-gray-600 text-sm">
          Get AI-powered stock price predictions using advanced Transformer neural networks
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 rounded-lg p-1 mb-6">
        <button
          onClick={() => setActiveTab('single')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'single'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Single Stock
        </button>
        <button
          onClick={() => setActiveTab('batch')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'batch'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Batch Analysis
        </button>
      </div>

      {activeTab === 'single' ? (
        <form onSubmit={handleSingleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Stock Symbol
            </label>
            <div className="relative">
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL"
                className="input-field pr-12"
                maxLength={5}
                disabled={isLoading}
              />
              <Search className="absolute right-3 top-3.5 w-4 h-4 text-gray-400" />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || !isApiHealthy}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="loading-spinner"></div>
                <span>Generating Prediction...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Get Prediction</span>
              </>
            )}
          </button>
        </form>
      ) : (
        <form onSubmit={handleBatchSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Add Stock Symbols ({batchSymbols.length}/10)
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={newBatchSymbol}
                onChange={(e) => setNewBatchSymbol(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL"
                className="input-field flex-1"
                maxLength={5}
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={addBatchSymbol}
                className="btn-secondary"
                disabled={isLoading}
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
          </div>

          {batchSymbols.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700">Selected Symbols:</p>
              <div className="flex flex-wrap gap-2">
                {batchSymbols.map((sym) => (
                  <div
                    key={sym}
                    className="flex items-center space-x-2 bg-blue-100 text-blue-800 px-3 py-1 rounded-lg text-sm"
                  >
                    <span className="font-medium">{sym}</span>
                    <button
                      type="button"
                      onClick={() => removeBatchSymbol(sym)}
                      className="hover:bg-blue-200 rounded-full p-1"
                      disabled={isLoading}
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading || !isApiHealthy || batchSymbols.length === 0}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="loading-spinner"></div>
                <span>Generating Predictions...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Get Batch Predictions</span>
              </>
            )}
          </button>
        </form>
      )}

      {/* Quick Actions */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-sm font-medium text-gray-700 mb-3">Popular Stocks:</p>
        <div className="grid grid-cols-3 gap-2">
          {popularSymbols.map((sym) => (
            <button
              key={sym}
              onClick={() => {
                if (activeTab === 'single') {
                  setSymbol(sym);
                } else {
                  setNewBatchSymbol(sym);
                }
              }}
              className="px-3 py-2 text-xs font-medium bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              disabled={isLoading}
            >
              {sym}
            </button>
          ))}
        </div>
      </div>

      {!isApiHealthy && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800 text-sm font-medium">
            ⚠️ API Connection Issue
          </p>
          <p className="text-red-600 text-xs mt-1">
            The prediction service is currently unavailable. Please check the API status above.
          </p>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
