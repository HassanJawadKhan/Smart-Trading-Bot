import React from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import PredictionResults from './components/PredictionResults';
import ApiStatus from './components/ApiStatus';
import ThemeProvider from './components/ThemeProvider';
import { usePredictions } from './hooks/usePredictions';
import { useApiHealth } from './hooks/useApiHealth';

function AppContent() {
  const predictions = usePredictions();
  const apiHealth = useApiHealth(true, 30000); // Check health every 30 seconds

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-gray-900">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          className: 'dark:!bg-slate-800/95 dark:!text-gray-200 dark:!border-slate-700/50',
          style: {
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '12px',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)',
          },
          success: {
            iconTheme: {
              primary: '#22c55e',
              secondary: '#ffffff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#ffffff',
            },
          },
        }}
      />

      <Header />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* API Status */}
        <ApiStatus 
          health={apiHealth.health}
          isLoading={apiHealth.isLoading}
          error={apiHealth.error}
          onRefresh={apiHealth.checkHealth}
        />

        {/* Dashboard with metrics */}
        <Dashboard 
          predictions={predictions.predictions}
          apiHealth={apiHealth}
        />

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Prediction Form */}
          <div className="lg:col-span-1">
            <PredictionForm
              onSinglePrediction={predictions.getSinglePrediction}
              onBatchPrediction={predictions.getBatchPredictions}
              isLoading={predictions.isLoading}
              supportedSymbols={apiHealth.supportedSymbols}
              isApiHealthy={apiHealth.isHealthy}
            />
          </div>

          {/* Prediction Results */}
          <div className="lg:col-span-2">
            <PredictionResults
              predictions={predictions.predictions}
              status={predictions.status}
              error={predictions.error}
              onClearPredictions={predictions.clearPredictions}
              onClearError={predictions.clearError}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center py-8 text-gray-500 dark:text-gray-400 text-sm">
          <div className="glass-card inline-block px-6 py-3 rounded-full">
            <p>
              Powered by <span className="gradient-text font-semibold">AI Transformer Neural Networks</span>
              {' '} • Built with React & TypeScript
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
