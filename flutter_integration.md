# Flutter App Integration Guide

## 🚀 Deploy the API Server

### Option 1: Local Testing
```bash
# Start the API server
cd /run/media/sofi/Study/Chat_Bot
source venv/bin/activate
python api_server.py
```
Server will run at: `http://localhost:8000`

### Option 2: Deploy to Production Server

#### Deploy to AWS EC2, Google Cloud, or DigitalOcean:
```bash
# 1. Install dependencies on server
pip install torch torchvision fastapi uvicorn pillow numpy python-multipart

# 2. Copy files to server
scp api_server.py dataset.py user@your-server:/app/

# 3. Run with systemd or PM2
uvicorn api_server:app --host 0.0.0.0 --port 8000

# 4. Use nginx as reverse proxy (optional)
# Configure nginx to forward requests to port 8000
```

#### Deploy to Heroku:
```bash
# Create Procfile
echo "web: uvicorn api_server:app --host 0.0.0.0 --port \$PORT" > Procfile

# Create requirements.txt
pip freeze > requirements.txt

# Deploy
heroku create your-app-name
git push heroku main
```

#### Deploy using Docker:
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py dataset.py ./

EXPOSE 8000

CMD ["python", "api_server.py"]
```

```bash
# Build and run
docker build -t object-detection-api .
docker run -p 8000:8000 object-detection-api
```

---

## 📱 Flutter App Integration

### Step 1: Add Dependencies to `pubspec.yaml`
```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.1.0
  image_picker: ^1.0.4
  dio: ^5.3.3  # Alternative HTTP client
```

### Step 2: Create API Service Class

```dart
// lib/services/object_detection_service.dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ObjectDetectionService {
  // Change this to your deployed server URL
  static const String baseUrl = 'http://YOUR_SERVER_IP:8000';
  
  // For local testing on Android emulator: http://10.0.2.2:8000
  // For local testing on iOS simulator: http://localhost:8000
  // For real device on same network: http://192.168.x.x:8000
  
  /// Check if API is healthy
  Future<bool> healthCheck() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }
  
  /// Get model information
  Future<Map<String, dynamic>?> getModelInfo() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/info'));
      if (response.statusCode == 200) {
        return json.decode(response.body);
      }
    } catch (e) {
      print('Get model info failed: $e');
    }
    return null;
  }
  
  /// Detect objects in image file
  Future<DetectionResult?> detectObjects(
    File imageFile, {
    double confidence = 0.2,
  }) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/detect'),
      );
      
      // Add image file
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );
      
      // Add confidence parameter
      request.fields['confidence'] = confidence.toString();
      
      // Send request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return DetectionResult.fromJson(data);
      } else {
        print('Detection failed: ${response.body}');
      }
    } catch (e) {
      print('Detection error: $e');
    }
    return null;
  }
  
  /// Detect objects using base64 encoded image
  Future<DetectionResult?> detectObjectsBase64(
    String base64Image, {
    double confidence = 0.2,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/detect_base64'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {
          'image_base64': base64Image,
          'confidence': confidence.toString(),
        },
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return DetectionResult.fromJson(data);
      } else {
        print('Detection failed: ${response.body}');
      }
    } catch (e) {
      print('Detection error: $e');
    }
    return null;
  }
}

/// Detection result model
class DetectionResult {
  final bool success;
  final int count;
  final List<Detection> detections;
  final double confidenceThreshold;
  
  DetectionResult({
    required this.success,
    required this.count,
    required this.detections,
    required this.confidenceThreshold,
  });
  
  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      success: json['success'] ?? false,
      count: json['count'] ?? 0,
      detections: (json['detections'] as List?)
          ?.map((e) => Detection.fromJson(e))
          .toList() ?? [],
      confidenceThreshold: json['confidence_threshold'] ?? 0.2,
    );
  }
}

/// Individual detection model
class Detection {
  final String className;
  final double confidence;
  final BoundingBox bbox;
  
  Detection({
    required this.className,
    required this.confidence,
    required this.bbox,
  });
  
  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      className: json['class'] ?? '',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      bbox: BoundingBox.fromJson(json['bbox']),
    );
  }
}

/// Bounding box model
class BoundingBox {
  final double x1, y1, x2, y2;
  
  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });
  
  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x1: (json['x1'] ?? 0.0).toDouble(),
      y1: (json['y1'] ?? 0.0).toDouble(),
      x2: (json['x2'] ?? 0.0).toDouble(),
      y2: (json['y2'] ?? 0.0).toDouble(),
    );
  }
  
  double get width => x2 - x1;
  double get height => y2 - y1;
}
```

### Step 3: Create UI Screen

```dart
// lib/screens/object_detection_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/object_detection_service.dart';

class ObjectDetectionScreen extends StatefulWidget {
  @override
  _ObjectDetectionScreenState createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  final ObjectDetectionService _service = ObjectDetectionService();
  final ImagePicker _picker = ImagePicker();
  
  File? _imageFile;
  DetectionResult? _result;
  bool _isLoading = false;
  double _confidence = 0.2;
  
  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _result = null;
      });
    }
  }
  
  Future<void> _detectObjects() async {
    if (_imageFile == null) return;
    
    setState(() {
      _isLoading = true;
    });
    
    final result = await _service.detectObjects(
      _imageFile!,
      confidence: _confidence,
    );
    
    setState(() {
      _result = result;
      _isLoading = false;
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Object Detection'),
        backgroundColor: Colors.blue,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image display
            if (_imageFile != null)
              Container(
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Image.file(_imageFile!, fit: BoxFit.contain),
              ),
            
            SizedBox(height: 16),
            
            // Image picker buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: Icon(Icons.camera_alt),
                    label: Text('Camera'),
                  ),
                ),
                SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: Icon(Icons.photo_library),
                    label: Text('Gallery'),
                  ),
                ),
              ],
            ),
            
            SizedBox(height: 16),
            
            // Confidence slider
            Text('Confidence Threshold: ${(_confidence * 100).toInt()}%'),
            Slider(
              value: _confidence,
              min: 0.05,
              max: 0.95,
              divisions: 18,
              label: '${(_confidence * 100).toInt()}%',
              onChanged: (value) {
                setState(() {
                  _confidence = value;
                });
              },
            ),
            
            SizedBox(height: 16),
            
            // Detect button
            ElevatedButton(
              onPressed: _imageFile != null && !_isLoading
                  ? _detectObjects
                  : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isLoading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text(
                      'DETECT OBJECTS',
                      style: TextStyle(fontSize: 18, color: Colors.white),
                    ),
            ),
            
            SizedBox(height: 24),
            
            // Results
            if (_result != null) ...[
              Text(
                'Detected ${_result!.count} objects:',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 12),
              
              ListView.builder(
                shrinkWrap: true,
                physics: NeverScrollableScrollPhysics(),
                itemCount: _result!.detections.length,
                itemBuilder: (context, index) {
                  final detection = _result!.detections[index];
                  return Card(
                    child: ListTile(
                      leading: CircleAvatar(
                        child: Text('${index + 1}'),
                      ),
                      title: Text(
                        detection.className,
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      subtitle: Text(
                        'Confidence: ${(detection.confidence * 100).toStringAsFixed(1)}%',
                      ),
                      trailing: Icon(Icons.check_circle, color: Colors.green),
                    ),
                  );
                },
              ),
            ],
          ],
        ),
      ),
    );
  }
}
```

### Step 4: Update AndroidManifest.xml

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<manifest>
    <!-- Add internet permission -->
    <uses-permission android:name="android.permission.INTERNET"/>
    <uses-permission android:name="android.permission.CAMERA"/>
    
    <!-- For cleartext traffic (HTTP instead of HTTPS) in development -->
    <application
        android:usesCleartextTraffic="true"
        ...>
    </application>
</manifest>
```

### Step 5: Update Info.plist for iOS

```xml
<!-- ios/Runner/Info.plist -->
<dict>
    <!-- Add camera permission -->
    <key>NSCameraUsageDescription</key>
    <string>We need camera access to detect objects</string>
    
    <key>NSPhotoLibraryUsageDescription</key>
    <string>We need photo library access to detect objects</string>
</dict>
```

---

## 🧪 Testing

### 1. Test API locally:
```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Test with curl
curl -X POST http://localhost:8000/detect \
  -F "file=@test_image.jpg" \
  -F "confidence=0.2"
```

### 2. Test from Flutter:
```dart
// For Android emulator
static const String baseUrl = 'http://10.0.2.2:8000';

// For iOS simulator
static const String baseUrl = 'http://localhost:8000';

// For real device (use your computer's IP)
static const String baseUrl = 'http://192.168.1.100:8000';
```

---

## 🌐 Production Deployment URLs

After deploying, update Flutter app with production URL:

```dart
// For AWS EC2
static const String baseUrl = 'http://ec2-xx-xx-xx-xx.compute.amazonaws.com:8000';

// For custom domain
static const String baseUrl = 'https://api.yourdomain.com';

// For Heroku
static const String baseUrl = 'https://your-app-name.herokuapp.com';
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/info` | GET | Model information |
| `/detect` | POST | Detect objects (multipart file) |
| `/detect_base64` | POST | Detect objects (base64 image) |

---

## 💡 Tips

1. **Use HTTPS in production** - Add SSL certificate
2. **Add authentication** - Use API keys or JWT tokens
3. **Rate limiting** - Prevent API abuse
4. **Caching** - Cache results for identical images
5. **Image optimization** - Compress images before sending
6. **Error handling** - Handle network errors gracefully
7. **Loading states** - Show progress indicators

---

## 🚀 Ready to Deploy!

1. Start API server: `python api_server.py`
2. Test API at: http://localhost:8000/docs
3. Update Flutter app with server URL
4. Build and run Flutter app
5. Upload image and see detections!
