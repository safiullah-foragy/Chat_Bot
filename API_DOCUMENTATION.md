# Object Detection Lite API

A lightweight REST API server for object detection that can be accessed from any application.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask flask-cors
```

### 2. Start the Server
```bash
python api_lite.py
```

The API will be available at: `http://localhost:5000`

## 📡 API Endpoints

### 1. **GET /** - API Information
Get basic information about the API.

**Example:**
```bash
curl http://localhost:5000/
```

### 2. **GET /health** - Health Check
Check if the API is running and model is loaded.

**Example:**
```bash
curl http://localhost:5000/health
```

### 3. **GET /classes** - List Classes
Get all detectable object classes (80 classes from COCO dataset).

**Example:**
```bash
curl http://localhost:5000/classes
```

### 4. **POST /detect** - Detect Objects (File Upload)
Upload an image file for object detection.

**Parameters:**
- `file`: Image file (multipart/form-data)
- `confidence`: Optional confidence threshold (default: 0.5)

**Example (curl):**
```bash
curl -X POST http://localhost:5000/detect \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

**Example (Python):**
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {'confidence': 0.5}
    response = requests.post('http://localhost:5000/detect', files=files, data=data)
    result = response.json()
    print(result)
```

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('confidence', '0.5');

fetch('http://localhost:5000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### 5. **POST /detect/url** - Detect Objects (Image URL)
Provide an image URL for object detection.

**Request Body (JSON):**
```json
{
  "image_url": "https://example.com/image.jpg",
  "confidence": 0.5
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5000/detect/url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg", "confidence": 0.5}'
```

**Example (Python):**
```python
import requests

payload = {
    "image_url": "https://example.com/image.jpg",
    "confidence": 0.5
}
response = requests.post('http://localhost:5000/detect/url', json=payload)
result = response.json()
print(result)
```

### 6. **POST /detect/base64** - Detect Objects (Base64 Image)
Send a base64 encoded image for object detection.

**Request Body (JSON):**
```json
{
  "image_base64": "base64_encoded_string_here",
  "confidence": 0.5
}
```

**Example (Python):**
```python
import requests
import base64

with open('image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "image_base64": image_base64,
    "confidence": 0.5
}
response = requests.post('http://localhost:5000/detect/base64', json=payload)
result = response.json()
print(result)
```

## 📦 Response Format

All detection endpoints return JSON in this format:

```json
{
  "success": true,
  "total_detections": 3,
  "confidence_threshold": 0.5,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "object": "dog",
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400
      },
      "center": {
        "x": 200,
        "y": 300
      },
      "size": {
        "width": 200,
        "height": 200
      }
    }
  ]
}
```

## 🔧 Integration Examples

### Flutter/Dart
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> detectObjects(File imageFile) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://your-server:5000/detect'),
  );
  
  request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
  request.fields['confidence'] = '0.5';
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  return json.decode(responseData);
}
```

### React Native
```javascript
const detectObjects = async (imageUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'photo.jpg',
  });
  formData.append('confidence', '0.5');
  
  const response = await fetch('http://your-server:5000/detect', {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return await response.json();
};
```

### Android (Java)
```java
OkHttpClient client = new OkHttpClient();

RequestBody requestBody = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "image.jpg",
        RequestBody.create(MediaType.parse("image/*"), imageFile))
    .addFormDataPart("confidence", "0.5")
    .build();

Request request = new Request.Builder()
    .url("http://your-server:5000/detect")
    .post(requestBody)
    .build();

Response response = client.newCall(request).execute();
String jsonResponse = response.body().string();
```

## 🌐 Deploy to Public Server

To make your API accessible from anywhere:

### Option 1: Use ngrok (Quick Testing)
```bash
# Install ngrok: https://ngrok.com/download
ngrok http 5000
```
You'll get a public URL like: `https://abc123.ngrok.io`

### Option 2: Deploy to Cloud
- **Heroku**: `git push heroku main`
- **Railway**: Connect GitHub repo
- **Render**: Deploy from GitHub
- **AWS/GCP/Azure**: Use container deployment

### Option 3: VPS/Dedicated Server
```bash
# Install and run with gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_lite:app
```

## 🔒 Security Notes

For production use:
1. Add API authentication (API keys or JWT)
2. Set up HTTPS/SSL
3. Add rate limiting
4. Validate file types and sizes
5. Use environment variables for configuration

## 🧪 Testing

Run the test script:
```bash
python test_api.py
```

## 📊 Performance

- Model: Faster R-CNN ResNet50 FPN
- Response time: ~1-3 seconds per image (CPU)
- Concurrent requests: Supports multiple simultaneous requests
- Image formats: JPEG, PNG, BMP, etc.

## 🆘 Troubleshooting

**Port already in use:**
```bash
# Change port in api_lite.py or kill existing process
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :5000
kill -9 <PID>
```

**CORS issues:**
- CORS is enabled by default for all origins
- Adjust in `api_lite.py` if needed

**Model loading slow:**
- First run downloads ~160MB model
- Cached for subsequent runs
