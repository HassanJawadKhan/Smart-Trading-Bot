import React from 'react';
import { AlertCircle, CheckCircle, RefreshCw, Server, Brain, Clock } from 'lucide-react';
import { HealthResponse } from '../types';
import { formatDateTime, formatRelativeTime } from '../utils';

interface ApiStatusProps {
  health: HealthResponse | null;
  isLoading: boolean;
  error: string | null;
  onRefresh: () => void;
}

const ApiStatus: React.FC<ApiStatusProps> = ({ health, isLoading, error, onRefresh }) => {
  const isHealthy = health?.status === 'healthy' && health?.model_loaded;
  
  return (
    <div className={`prediction-card border-l-4 ${
      error ? 'border-l-red-500 bg-red-50/30 dark:bg-red-900/10' : 
      isHealthy ? 'border-l-green-500 bg-green-50/30 dark:bg-green-900/10' : 
      'border-l-yellow-500 bg-yellow-50/30 dark:bg-yellow-900/10'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          {error ? (
            <AlertCircle className="w-6 h-6 text-red-500" />
          ) : isHealthy ? (
            <CheckCircle className="w-6 h-6 text-green-500" />
          ) : (
            <Server className="w-6 h-6 text-yellow-500" />
          )}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              API Status
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {error ? 'Connection Failed' : 
               isHealthy ? 'All Systems Operational' : 
               'Checking Status...'}
            </p>
          </div>
        </div>
        
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="btn-secondary flex items-center space-x-2 text-sm"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          <span>{isLoading ? 'Checking...' : 'Refresh'}</span>
        </button>
      </div>

      {error ? (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/50 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-400 font-medium">Connection Error</p>
          <p className="text-red-600 dark:text-red-300 text-sm mt-1">{error}</p>
        </div>
      ) : health ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="metric-card">
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-blue-500" />
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">AI Model</p>
                <p className={`text-sm ${health.model_loaded ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {health.model_loaded ? 'Loaded & Ready' : 'Not Available'}
                </p>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="flex items-center space-x-2">
              <Server className="w-5 h-5 text-indigo-500" />
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">Features</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {health.model_info.features} Technical Indicators
                </p>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-purple-500" />
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">Last Updated</p>
                <p className="text-sm text-gray-600 dark:text-gray-400" title={formatDateTime(health.timestamp)}>
                  {formatRelativeTime(health.timestamp)}
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center py-8">
          <div className="loading-spinner mr-3"></div>
          <span className="text-gray-600 dark:text-gray-400">Connecting to API...</span>
        </div>
      )}

      {health && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
            <span>Supported Symbols: {health.supported_symbols.length}</span>
            <span>Device: {health.model_info.device.toUpperCase()}</span>
            <span>Version: v1.0.0</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ApiStatus;
