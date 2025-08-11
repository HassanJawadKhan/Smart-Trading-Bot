#!/usr/bin/env python3
"""
Simple test script for the Stock Prediction API
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_api():
    """Test the main API endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("Testing Stock Prediction API...")
        print("=" * 50)
        
        # Test root endpoint
        print("1. Testing root endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/")
            print(f"✓ Status: {response.status_code}")
            print(f"✓ Response: {response.json()}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n" + "-" * 30)
        
        # Test health endpoint
        print("2. Testing health endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"✓ Status: {response.status_code}")
            data = response.json()
            print(f"✓ Model loaded: {data.get('model_loaded', False)}")
            print(f"✓ Supported symbols: {len(data.get('supported_symbols', []))}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n" + "-" * 30)
        
        # Test single prediction
        print("3. Testing single stock prediction...")
        try:
            prediction_data = {
                "symbol": "AAPL",
                "days_ahead": 1
            }
            response = await client.post(f"{BASE_URL}/predict", json=prediction_data)
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Symbol: {data['symbol']}")
                print(f"✓ Current Price: ${data['current_price']}")
                print(f"✓ Predicted Price: ${data['predicted_price']}")
                print(f"✓ Change: {data['predicted_change_percent']:.2f}%")
                print(f"✓ Confidence: {data['confidence_score']:.2f}")
            else:
                print(f"✗ Error response: {response.text}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n" + "-" * 30)
        
        # Test batch prediction
        print("4. Testing batch prediction...")
        try:
            batch_data = {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "days_ahead": 1
            }
            response = await client.post(f"{BASE_URL}/batch-predict", json=batch_data)
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Successfully predicted {len(data)} stocks")
                for prediction in data[:2]:  # Show first 2 results
                    print(f"  - {prediction['symbol']}: ${prediction['current_price']} → ${prediction['predicted_price']} ({prediction['predicted_change_percent']:+.2f}%)")
            else:
                print(f"✗ Error response: {response.text}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n" + "-" * 30)
        
        # Test supported symbols endpoint
        print("5. Testing supported symbols...")
        try:
            response = await client.get(f"{BASE_URL}/supported-symbols")
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                symbols = response.json()
                print(f"✓ Supported symbols ({len(symbols)}): {symbols[:10]}...")  # Show first 10
            else:
                print(f"✗ Error response: {response.text}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n" + "=" * 50)
        print("API testing complete!")

if __name__ == "__main__":
    asyncio.run(test_api())
