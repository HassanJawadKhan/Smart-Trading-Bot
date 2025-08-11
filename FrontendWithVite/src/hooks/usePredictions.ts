import { useState, useCallback } from 'react';
import { PredictionRequest, BatchPredictionRequest, PredictionResponse, PredictionStatus } from '../types';
import { StockPredictionAPI } from '../services/api';
import { generateId } from '../utils';

interface UsePredictionsReturn {
  predictions: PredictionResponse[];
  status: PredictionStatus;
  error: string | null;
  isLoading: boolean;
  getSinglePrediction: (request: PredictionRequest) => Promise<PredictionResponse | null>;
  getBatchPredictions: (request: BatchPredictionRequest) => Promise<PredictionResponse[] | null>;
  clearPredictions: () => void;
  clearError: () => void;
}

export function usePredictions(): UsePredictionsReturn {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [status, setStatus] = useState<PredictionStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const isLoading = status === 'loading';

  const getSinglePrediction = useCallback(async (request: PredictionRequest): Promise<PredictionResponse | null> => {
    setStatus('loading');
    setError(null);

    try {
      const response = await StockPredictionAPI.getPrediction(request);
      
      if (response.error) {
        setError(response.error);
        setStatus('error');
        return null;
      }

      if (response.data) {
        setPredictions(prev => {
          // Remove any existing prediction for the same symbol
          const filtered = prev.filter(p => p.symbol !== response.data!.symbol);
          return [response.data!, ...filtered];
        });
        setStatus('success');
        return response.data;
      }

      return null;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get prediction';
      setError(errorMessage);
      setStatus('error');
      return null;
    }
  }, []);

  const getBatchPredictions = useCallback(async (request: BatchPredictionRequest): Promise<PredictionResponse[] | null> => {
    setStatus('loading');
    setError(null);

    try {
      const response = await StockPredictionAPI.getBatchPredictions(request);
      
      if (response.error) {
        setError(response.error);
        setStatus('error');
        return null;
      }

      if (response.data) {
        setPredictions(prev => {
          const existingSymbols = new Set(prev.map(p => p.symbol));
          const newPredictions = response.data!.filter(p => !existingSymbols.has(p.symbol));
          return [...newPredictions, ...prev];
        });
        setStatus('success');
        return response.data;
      }

      return null;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get batch predictions';
      setError(errorMessage);
      setStatus('error');
      return null;
    }
  }, []);

  const clearPredictions = useCallback(() => {
    setPredictions([]);
    setStatus('idle');
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    predictions,
    status,
    error,
    isLoading,
    getSinglePrediction,
    getBatchPredictions,
    clearPredictions,
    clearError,
  };
}
