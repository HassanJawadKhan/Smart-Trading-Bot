#!/usr/bin/env python3
"""
Startup script for Stock Prediction API
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Start the FastAPI application"""
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    print("🚀 Starting Stock Prediction API...")
    print(f"📂 Working directory: {current_dir}")
    print("🔗 API will be available at:")
    print("   - Main API: http://localhost:8000")
    print("   - Documentation: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("   - Health Check: http://localhost:8000/health")
    print("\n" + "="*50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
