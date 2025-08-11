import React from 'react';
import { BarChart3, TrendingUp, TrendingDown, Target, Users, Clock } from 'lucide-react';
import { PredictionResponse } from '../types';
import { formatCurrency, formatPercentage, calculateAccuracy } from '../utils';

interface DashboardProps {
  predictions: PredictionResponse[];
  apiHealth: any;
}

const Dashboard: React.FC<DashboardProps> = ({ predictions, apiHealth }) => {
  const totalPredictions = predictions.length;
  const bullishPredictions = predictions.filter(p => p.predicted_change_percent > 0).length;
  const bearishPredictions = predictions.filter(p => p.predicted_change_percent < 0).length;
  const avgConfidence = predictions.length > 0 
    ? predictions.reduce((sum, p) => sum + p.confidence_score, 0) / predictions.length 
    : 0;
  
  const avgChange = predictions.length > 0
    ? predictions.reduce((sum, p) => sum + p.predicted_change_percent, 0) / predictions.length
    : 0;

  const highestGainer = predictions.length > 0
    ? predictions.reduce((max, p) => p.predicted_change_percent > max.predicted_change_percent ? p : max)
    : null;

  const metrics = [
    {
      title: 'Total Predictions',
      value: totalPredictions,
      icon: BarChart3,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      change: '+12%',
    },
    {
      title: 'Model Accuracy',
      value: `${(calculateAccuracy(predictions) * 100).toFixed(1)}%`,
      icon: Target,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      change: '+2.3%',
    },
    {
      title: 'Avg Confidence',
      value: `${(avgConfidence * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
      change: '+5.1%',
    },
    {
      title: 'Supported Symbols',
      value: apiHealth.supportedSymbols.length,
      icon: Users,
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-100',
      change: 'Stable',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Main Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div key={index} className="prediction-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">
                    {metric.title}
                  </p>
                  <p className="text-2xl font-bold text-gray-900">
                    {metric.value}
                  </p>
                  <div className="flex items-center mt-2">
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      metric.change.startsWith('+') 
                        ? 'bg-green-100 text-green-800'
                        : metric.change === 'Stable'
                        ? 'bg-gray-100 text-gray-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {metric.change}
                    </span>
                  </div>
                </div>
                <div className={`p-3 rounded-xl ${metric.bgColor}`}>
                  <Icon className={`w-6 h-6 ${metric.color}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Prediction Summary */}
      {predictions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="prediction-card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Prediction Distribution
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium">Bullish Signals</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-600">{bullishPredictions}</span>
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${totalPredictions > 0 ? (bullishPredictions / totalPredictions) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <TrendingDown className="w-4 h-4 text-red-500" />
                  <span className="text-sm font-medium">Bearish Signals</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-600">{bearishPredictions}</span>
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${totalPredictions > 0 ? (bearishPredictions / totalPredictions) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="prediction-card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Market Overview
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Average Change</span>
                <span className={`font-semibold ${avgChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatPercentage(avgChange)}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Best Performer</span>
                <div className="text-right">
                  {highestGainer ? (
                    <>
                      <div className="font-semibold text-gray-900">{highestGainer.symbol}</div>
                      <div className="text-sm text-green-600">
                        {formatPercentage(highestGainer.predicted_change_percent)}
                      </div>
                    </>
                  ) : (
                    <span className="text-gray-500">No data</span>
                  )}
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Model Status</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-green-600">Active</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {predictions.length === 0 && (
        <div className="prediction-card text-center py-12">
          <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            No Predictions Yet
          </h3>
          <p className="text-gray-600 max-w-md mx-auto">
            Start by entering a stock symbol in the prediction form to see AI-powered analysis and forecasts.
          </p>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
