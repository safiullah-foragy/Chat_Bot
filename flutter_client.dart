// Object Detection API Client for Flutter
// Add to your Flutter project

import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

/// Object Detection API Client
class ObjectDetectionAPI {
  // Replace with your Render.com URL after deployment
  static const String baseUrl = 'https://your-app-name.onrender.com';
  
  // Or use localhost for testing
  // static const String baseUrl = 'http://10.0.2.2:5000'; // Android emulator
  // static const String baseUrl = 'http://localhost:5000'; // iOS simulator
  
  /// Check if API is healthy
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 10));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['status'] == 'healthy';
      }
      return false;
    } catch (e) {
      print('Health check error: $e');
      return false;
    }
  }
  
  /// Get list of detectable classes
  static Future<List<String>?> getClasses() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/classes'),
      ).timeout(const Duration(seconds: 10));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return List<String>.from(data['classes']);
      }
      return null;
    } catch (e) {
      print('Get classes error: $e');
      return null;
    }
  }
  
  /// Detect objects in image file
  static Future<DetectionResult?> detectObjects({
    required File imageFile,
    double confidence = 0.5,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/detect'),
      );
      
      // Add image file
      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
          contentType: MediaType('image', 'jpeg'),
        ),
      );
      
      // Add confidence threshold
      request.fields['confidence'] = confidence.toString();
      
      // Send request with longer timeout for first request (cold start)
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 60),
      );
      
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return DetectionResult.fromJson(data);
      } else {
        print('Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('Detection error: $e');
      return null;
    }
  }
  
  /// Detect objects from image URL
  static Future<DetectionResult?> detectFromUrl({
    required String imageUrl,
    double confidence = 0.5,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/detect/url'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'image_url': imageUrl,
          'confidence': confidence,
        }),
      ).timeout(const Duration(seconds: 60));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return DetectionResult.fromJson(data);
      } else {
        print('Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('Detection from URL error: $e');
      return null;
    }
  }
  
  /// Detect objects from base64 encoded image
  static Future<DetectionResult?> detectFromBase64({
    required String imageBase64,
    double confidence = 0.5,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/detect/base64'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'image_base64': imageBase64,
          'confidence': confidence,
        }),
      ).timeout(const Duration(seconds: 60));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return DetectionResult.fromJson(data);
      } else {
        print('Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('Detection from base64 error: $e');
      return null;
    }
  }
}

/// Detection Result Model
class DetectionResult {
  final bool success;
  final int totalDetections;
  final double confidenceThreshold;
  final ImageSize imageSize;
  final List<Detection> detections;
  
  DetectionResult({
    required this.success,
    required this.totalDetections,
    required this.confidenceThreshold,
    required this.imageSize,
    required this.detections,
  });
  
  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      success: json['success'] ?? false,
      totalDetections: json['total_detections'] ?? 0,
      confidenceThreshold: (json['confidence_threshold'] ?? 0.5).toDouble(),
      imageSize: ImageSize.fromJson(json['image_size'] ?? {}),
      detections: (json['detections'] as List? ?? [])
          .map((d) => Detection.fromJson(d))
          .toList(),
    );
  }
}

/// Image Size Model
class ImageSize {
  final int width;
  final int height;
  
  ImageSize({required this.width, required this.height});
  
  factory ImageSize.fromJson(Map<String, dynamic> json) {
    return ImageSize(
      width: json['width'] ?? 0,
      height: json['height'] ?? 0,
    );
  }
}

/// Detection Model
class Detection {
  final String object;
  final double confidence;
  final BoundingBox bbox;
  final Point center;
  final Size size;
  
  Detection({
    required this.object,
    required this.confidence,
    required this.bbox,
    required this.center,
    required this.size,
  });
  
  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      object: json['object'] ?? '',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      bbox: BoundingBox.fromJson(json['bbox'] ?? {}),
      center: Point.fromJson(json['center'] ?? {}),
      size: Size.fromJson(json['size'] ?? {}),
    );
  }
}

/// Bounding Box Model
class BoundingBox {
  final int x1;
  final int y1;
  final int x2;
  final int y2;
  
  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });
  
  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x1: json['x1'] ?? 0,
      y1: json['y1'] ?? 0,
      x2: json['x2'] ?? 0,
      y2: json['y2'] ?? 0,
    );
  }
}

/// Point Model
class Point {
  final int x;
  final int y;
  
  Point({required this.x, required this.y});
  
  factory Point.fromJson(Map<String, dynamic> json) {
    return Point(
      x: json['x'] ?? 0,
      y: json['y'] ?? 0,
    );
  }
}

/// Size Model
class Size {
  final int width;
  final int height;
  
  Size({required this.width, required this.height});
  
  factory Size.fromJson(Map<String, dynamic> json) {
    return Size(
      width: json['width'] ?? 0,
      height: json['height'] ?? 0,
    );
  }
}
