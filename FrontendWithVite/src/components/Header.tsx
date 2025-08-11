import React from 'react';
import { Brain, TrendingUp, Zap } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

const Header: React.FC = () => {
  return (
    <header className="glass-card sticky top-0 z-50 border-b border-white/20">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl blur opacity-75"></div>
              <div className="relative bg-gradient-to-r from-blue-600 to-indigo-600 p-3 rounded-xl">
                <Brain className="w-8 h-8 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">
                Smart Stock Predictor
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Professional Trading Analytics
              </p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer">
              <TrendingUp className="w-4 h-4" />
              <span className="font-medium">Predictions</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer">
              <Zap className="w-4 h-4" />
              <span className="font-medium">Analytics</span>
            </div>
          </nav>

          {/* Right Side Actions */}
          <div className="flex items-center space-x-4">
            {/* Status Indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">AI Model Active</span>
            </div>
            
            {/* Theme Toggle */}
            <ThemeToggle />
            
            {/* Mobile menu button (placeholder) */}
            <button className="md:hidden p-2 rounded-lg hover:bg-white/50 dark:hover:bg-slate-700/50 transition-colors">
              <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Subtitle */}
        <div className="mt-2 text-center text-sm text-gray-500 dark:text-gray-400 hidden sm:block">
          Advanced Transformer Neural Networks • Real-time Predictions • Professional Grade Analytics
        </div>
      </div>
    </header>
  );
};

export default Header;
