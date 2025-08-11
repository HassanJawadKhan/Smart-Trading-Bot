import axios, { AxiosResponse } from 'axios';
import {
  PredictionRequest,
  BatchPredictionRequest,
  PredictionResponse,
  HealthResponse,
  ApiResponse,
} from '../types';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for consistent error handling
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    const errorMessage = error.response?.data?.detail || error.message || 'An unexpected error occurred';
    return Promise.reject(new Error(errorMessage));
  }
);

export class StockPredictionAPI {
  /**
   * Get API health status and model information
   */
  static async getHealth(): Promise<ApiResponse<HealthResponse>> {
    try {
      const response = await api.get<HealthResponse>('/health');
      return { data: response.data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Failed to get health status' };
    }
  }

  /**
   * Get single stock prediction
   */
  static async getPrediction(request: PredictionRequest): Promise<ApiResponse<PredictionResponse>> {
    try {
      const response = await api.post<PredictionResponse>('/predict', request);
      return { data: response.data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Failed to get prediction' };
    }
  }

  /**
   * Get batch predictions for multiple stocks
   */
  static async getBatchPredictions(
    request: BatchPredictionRequest
  ): Promise<ApiResponse<PredictionResponse[]>> {
    try {
      const response = await api.post<PredictionResponse[]>('/batch-predict', request);
      return { data: response.data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Failed to get batch predictions' };
    }
  }

  /**
   * Get supported symbols
   */
  static async getSupportedSymbols(): Promise<ApiResponse<string[]>> {
    try {
      const response = await api.get<string[]>('/supported-symbols');
      return { data: response.data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Failed to get supported symbols' };
    }
  }

  /**
   * Get API root information
   */
  static async getApiInfo(): Promise<ApiResponse<any>> {
    try {
      const response = await api.get('/');
      return { data: response.data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Failed to get API info' };
    }
  }

  /**
   * Reload the AI model
   */
  static async reloadModel(): Promise<ApiResponse<{ status: string; message: string }>> {
    try {
      const response = await api.post<{ status: string; message: string }>('/reload-model');
      return { data: response.data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Failed to reload model' };
    }
  }
}

export default StockPredictionAPI;
