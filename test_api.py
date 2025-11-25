"""
Test script for API Lite Server
Shows how to call the API from your app
"""

import requests
import json

# API base URL
API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_classes():
    """Test classes endpoint"""
    print("Testing /classes endpoint...")
    response = requests.get(f"{API_URL}/classes")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total classes: {data['total_classes']}")
    print(f"Sample classes: {data['classes'][:10]}")
    print()

def test_detect_file():
    """Test detection with file upload"""
    print("Testing /detect endpoint with file upload...")
    
    # Upload image file
    with open('test_dog.jpg', 'rb') as f:
        files = {'file': f}
        data = {'confidence': 0.5}
        response = requests.post(f"{API_URL}/detect", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result['success']:
        print(f"✅ Detected {result['total_detections']} objects:")
        for i, det in enumerate(result['detections'][:5], 1):
            print(f"  {i}. {det['object']} - {det['confidence']:.2%}")
    else:
        print(f"❌ Error: {result['error']}")
    print()

def test_detect_url():
    """Test detection with image URL"""
    print("Testing /detect/url endpoint...")
    
    payload = {
        "image_url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1",
        "confidence": 0.5
    }
    
    response = requests.post(f"{API_URL}/detect/url", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result['success']:
        print(f"✅ Detected {result['total_detections']} objects:")
        for i, det in enumerate(result['detections'][:5], 1):
            print(f"  {i}. {det['object']} - {det['confidence']:.2%}")
    else:
        print(f"❌ Error: {result['error']}")
    print()

def test_detect_base64():
    """Test detection with base64 image"""
    print("Testing /detect/base64 endpoint...")
    
    import base64
    
    # Read image and convert to base64
    with open('test_dog.jpg', 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "image_base64": image_data,
        "confidence": 0.5
    }
    
    response = requests.post(f"{API_URL}/detect/base64", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result['success']:
        print(f"✅ Detected {result['total_detections']} objects:")
        for i, det in enumerate(result['detections'][:5], 1):
            print(f"  {i}. {det['object']} - {det['confidence']:.2%}")
    else:
        print(f"❌ Error: {result['error']}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("API LITE SERVER TEST")
    print("=" * 70)
    print()
    
    try:
        # Test all endpoints
        test_health()
        test_classes()
        test_detect_file()
        # test_detect_url()  # Uncomment to test with URL
        # test_detect_base64()  # Uncomment to test with base64
        
        print("=" * 70)
        print("✅ All tests completed!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("Make sure the server is running: python api_lite.py")
    except Exception as e:
        print(f"❌ Error: {e}")
