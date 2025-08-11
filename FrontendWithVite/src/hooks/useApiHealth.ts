import { useState, useEffect, useCallback } from 'react';
import { HealthResponse } from '../types';
import { StockPredictionAPI } from '../services/api';

interface UseApiHealthReturn {
  health: HealthResponse | null;
  isLoading: boolean;
  error: string | null;
  isHealthy: boolean;
  supportedSymbols: string[];
  checkHealth: () => Promise<void>;
  lastChecked: Date | null;
}

export function useApiHealth(autoCheck: boolean = true, interval: number = 30000): UseApiHealthReturn {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const checkHealth = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await StockPredictionAPI.getHealth();
      
      if (response.error) {
        setError(response.error);
        setHealth(null);
      } else if (response.data) {
        setHealth(response.data);
        setError(null);
      }
      
      setLastChecked(new Date());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to check API health';
      setError(errorMessage);
      setHealth(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial health check
  useEffect(() => {
    if (autoCheck) {
      checkHealth();
    }
  }, [checkHealth, autoCheck]);

  // Periodic health checks
  useEffect(() => {
    if (!autoCheck || interval <= 0) return;

    const intervalId = setInterval(checkHealth, interval);
    return () => clearInterval(intervalId);
  }, [checkHealth, autoCheck, interval]);

  const isHealthy = health?.status === 'healthy' && health?.model_loaded === true;
  const supportedSymbols = health?.supported_symbols || [];

  return {
    health,
    isLoading,
    error,
    isHealthy,
    supportedSymbols,
    checkHealth,
    lastChecked,
  };
}
