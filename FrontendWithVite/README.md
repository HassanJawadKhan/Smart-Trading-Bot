# AI Stock Prediction Frontend

A modern, professional React TypeScript frontend for the AI Stock Prediction API. Built with Vite, Tailwind CSS, and featuring a beautiful glassmorphism UI design.

## Features

- 🤖 **AI-Powered Predictions**: Real-time stock price predictions using Transformer neural networks
- 🎨 **Modern UI**: Beautiful glassmorphism design with smooth animations
- 📱 **Responsive**: Fully responsive design that works on all devices
- 🚀 **Fast**: Built with Vite for lightning-fast development and builds
- 📊 **Interactive Dashboard**: Comprehensive metrics and analytics
- 🔥 **Real-time**: Live API health monitoring and status updates
- 💾 **Batch Processing**: Support for multiple stock predictions simultaneously
- 🎯 **TypeScript**: Full type safety and excellent developer experience

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with custom glassmorphism components
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Charts**: Recharts (for future enhancements)
- **Notifications**: React Hot Toast
- **State Management**: React Hooks (useState, useEffect, custom hooks)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Running Backend API (see ../backend)

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser to http://localhost:3000

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── components/          # React components
│   ├── Header.tsx      # Navigation header
│   ├── Dashboard.tsx   # Metrics dashboard
│   ├── ApiStatus.tsx   # API health monitoring
│   ├── PredictionForm.tsx    # Input form
│   └── PredictionResults.tsx # Results display
├── hooks/              # Custom React hooks
│   ├── usePredictions.ts     # Prediction management
│   └── useApiHealth.ts       # API health monitoring
├── services/           # API services
│   └── api.ts         # Backend API client
├── types/             # TypeScript type definitions
│   └── index.ts       # Shared types
├── utils/             # Utility functions
│   └── index.ts       # Helper functions
├── App.tsx            # Main app component
├── main.tsx           # Entry point
└── index.css          # Global styles with Tailwind
```

## API Integration

The frontend integrates with the FastAPI backend through a comprehensive API service:

### Endpoints Used
- `GET /health` - API health and model status
- `POST /predict` - Single stock prediction
- `POST /batch-predict` - Multiple stock predictions
- `GET /supported-symbols` - Available stock symbols

### Features
- Automatic error handling and retries
- Loading states and user feedback
- Real-time health monitoring
- Batch processing support

## Components

### Dashboard
- Real-time metrics and statistics
- Prediction distribution charts
- Model performance indicators
- Market overview

### PredictionForm
- Single stock prediction input
- Batch prediction with symbol management
- Popular stocks quick-select
- Input validation and error handling

### PredictionResults
- Beautiful card-based results display
- Confidence score indicators
- Price change visualization
- Timestamp and metadata

### ApiStatus
- Real-time connection monitoring
- Model status and information
- Health check controls
- Error state handling

## Customization

### Styling
The app uses Tailwind CSS with custom components defined in `index.css`:

- `.glass-card` - Glassmorphism card effect
- `.gradient-text` - Gradient text styling
- `.btn-primary` - Primary button styling
- `.prediction-card` - Prediction result cards

### Configuration
Key configuration in `vite.config.ts`:
- API proxy to backend (port 8000)
- Development server (port 3000)
- Build optimizations

## Performance

- **Lazy Loading**: Components load on demand
- **Optimized Builds**: Vite's optimized bundling
- **Efficient State**: Custom hooks for state management
- **Debounced Inputs**: Reduced API calls
- **Memoization**: Prevented unnecessary re-renders

## Future Enhancements

- [ ] Historical prediction tracking
- [ ] Advanced charting with Recharts
- [ ] Dark mode support
- [ ] Export predictions to CSV/PDF
- [ ] Real-time WebSocket updates
- [ ] Mobile app with React Native
- [ ] Advanced filtering and sorting
- [ ] User preferences and settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the AI Stock Prediction system and follows the same licensing terms.

---

Built with ❤️ using React, TypeScript, and Tailwind CSS
