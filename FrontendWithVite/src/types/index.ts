export interface PredictionRequest {
  symbol: string;
  days_ahead?: number;
}

export interface BatchPredictionRequest {
  symbols: string[];
  days_ahead?: number;
}

export interface PredictionResponse {
  symbol: string;
  current_price: number;
  predicted_price: number;
  predicted_change_percent: number;
  confidence_score: number;
  days_ahead: number;
  timestamp: string;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  supported_symbols: string[];
  model_info: {
    features: number;
    training_date: string;
    device: string;
  };
  timestamp: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
}

export interface StockSymbol {
  symbol: string;
  name?: string;
  sector?: string;
}

export interface PredictionHistory {
  id: string;
  symbol: string;
  predictions: PredictionResponse[];
  createdAt: string;
}

export interface ChartData {
  name: string;
  current: number;
  predicted: number;
  change: number;
}

export type PredictionStatus = 'idle' | 'loading' | 'success' | 'error';

export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface ErrorState {
  hasError: boolean;
  message?: string;
  details?: string;
}
