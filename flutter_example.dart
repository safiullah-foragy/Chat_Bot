// Flutter Example: How to use Object Detection API
// Complete working example

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'flutter_client.dart'; // Import the API client

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const ObjectDetectionScreen(),
    );
  }
}

class ObjectDetectionScreen extends StatefulWidget {
  const ObjectDetectionScreen({Key? key}) : super(key: key);

  @override
  State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  File? _imageFile;
  DetectionResult? _result;
  bool _isLoading = false;
  bool _isHealthy = false;
  String _statusMessage = '';
  
  final ImagePicker _picker = ImagePicker();
  
  @override
  void initState() {
    super.initState();
    _checkHealth();
  }
  
  /// Check API health on startup
  Future<void> _checkHealth() async {
    setState(() {
      _statusMessage = 'Checking API status...';
    });
    
    final isHealthy = await ObjectDetectionAPI.checkHealth();
    
    setState(() {
      _isHealthy = isHealthy;
      _statusMessage = isHealthy 
          ? '✅ API is ready!' 
          : '⚠️ API not available (sleeping or offline)';
    });
    
    if (!isHealthy) {
      _showSnackBar(
        'API might be sleeping. First request may take 50+ seconds.',
        isError: true,
      );
    }
  }
  
  /// Pick image from gallery
  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1080,
      );
      
      if (image != null) {
        setState(() {
          _imageFile = File(image.path);
          _result = null;
        });
      }
    } catch (e) {
      _showSnackBar('Error picking image: $e', isError: true);
    }
  }
  
  /// Take photo with camera
  Future<void> _takePhoto() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1920,
        maxHeight: 1080,
      );
      
      if (image != null) {
        setState(() {
          _imageFile = File(image.path);
          _result = null;
        });
      }
    } catch (e) {
      _showSnackBar('Error taking photo: $e', isError: true);
    }
  }
  
  /// Detect objects in selected image
  Future<void> _detectObjects() async {
    if (_imageFile == null) {
      _showSnackBar('Please select an image first', isError: true);
      return;
    }
    
    setState(() {
      _isLoading = true;
      _result = null;
      _statusMessage = 'Detecting objects...';
    });
    
    try {
      final result = await ObjectDetectionAPI.detectObjects(
        imageFile: _imageFile!,
        confidence: 0.5,
      );
      
      setState(() {
        _result = result;
        _isLoading = false;
        _statusMessage = result != null
            ? '✅ Found ${result.totalDetections} objects!'
            : '❌ Detection failed';
      });
      
      if (result == null) {
        _showSnackBar('Detection failed. Check API status.', isError: true);
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = '❌ Error: $e';
      });
      _showSnackBar('Error: $e', isError: true);
    }
  }
  
  /// Show snackbar message
  void _showSnackBar(String message, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Colors.red : Colors.green,
        duration: const Duration(seconds: 3),
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Object Detection'),
        actions: [
          IconButton(
            icon: Icon(
              _isHealthy ? Icons.check_circle : Icons.error,
              color: _isHealthy ? Colors.green : Colors.orange,
            ),
            onPressed: _checkHealth,
            tooltip: 'Check API Status',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Card
            Card(
              color: _isHealthy ? Colors.green.shade50 : Colors.orange.shade50,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  _statusMessage,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Image Display
            if (_imageFile != null)
              Card(
                clipBehavior: Clip.antiAlias,
                child: Image.file(
                  _imageFile!,
                  height: 300,
                  fit: BoxFit.cover,
                ),
              )
            else
              Card(
                child: Container(
                  height: 300,
                  color: Colors.grey.shade200,
                  child: const Center(
                    child: Icon(
                      Icons.image,
                      size: 64,
                      color: Colors.grey,
                    ),
                  ),
                ),
              ),
            
            const SizedBox(height: 16),
            
            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _pickImage,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _takePhoto,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 8),
            
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _detectObjects,
              icon: _isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.search),
              label: Text(_isLoading ? 'Detecting...' : 'Detect Objects'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(16),
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Results
            if (_result != null && _result!.success)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '🎯 Detected ${_result!.totalDetections} Objects',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const Divider(height: 24),
                      if (_result!.detections.isEmpty)
                        const Text(
                          'No objects detected. Try lowering the confidence threshold.',
                          style: TextStyle(color: Colors.grey),
                        )
                      else
                        ListView.builder(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          itemCount: _result!.detections.length,
                          itemBuilder: (context, index) {
                            final detection = _result!.detections[index];
                            return ListTile(
                              leading: CircleAvatar(
                                backgroundColor: Colors.blue,
                                child: Text('${index + 1}'),
                              ),
                              title: Text(
                                detection.object.toUpperCase(),
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              subtitle: Text(
                                'Confidence: ${(detection.confidence * 100).toStringAsFixed(1)}%\n'
                                'Position: (${detection.center.x}, ${detection.center.y})\n'
                                'Size: ${detection.size.width}×${detection.size.height}',
                              ),
                              isThreeLine: true,
                            );
                          },
                        ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
