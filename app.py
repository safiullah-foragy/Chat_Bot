"""
Object Detection Web UI with Gradio
Upload an image and get detailed object detection results
"""

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

from model import get_model
from dataset import COCO_CLASSES
from feedback_db import feedback_db


# Global variables
model = None
device = None
COLORS = None


def initialize_model():
    """Initialize the pretrained model with upgraded architecture"""
    global model, device, COLORS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # UPGRADED: Keep V1 but optimize with better settings
    # V2 download is unstable, keeping reliable V1 with optimizations
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    
    print("Loading Faster R-CNN ResNet50 FPN V1 (OPTIMIZED)...")
    print("🚀 Using DEFAULT weights + optimized inference settings!")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # Uses best available V1 weights
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    
    # OPTIMIZATION: Set model to use better NMS settings
    model.roi_heads.score_thresh = 0.05  # Lower threshold for more detections
    model.roi_heads.nms_thresh = 0.3  # Stricter NMS to reduce duplicates
    
    print(f"✅ Model loaded: Faster R-CNN ResNet50 FPN V1 (OPTIMIZED)")
    print(f"📊 Training data: COCO dataset (330,000+ images, 1.5M+ instances, 80 classes)")
    print(f"💡 Optimizations: Lower threshold (0.05) + Better NMS (0.3)")
    print(f"📈 Feedback system: Collecting user data for future fine-tuning")
    
    # Generate colors for visualization
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)
    
    print("Model loaded successfully!")


def predict_objects(image, confidence_threshold):
    """
    Predict objects in an image
    
    Args:
        image: PIL Image or numpy array
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        annotated_image: Image with bounding boxes
        results_text: Formatted detection results
        results_json: JSON string of detections
    """
    if image is None:
        return None, "Please upload an image first.", "{}"
    
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Preprocess image
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Extract results
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Create annotated image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Prepare results
    detections = []
    results_lines = []
    
    if len(boxes) == 0:
        results_text = f"No objects detected with confidence >= {confidence_threshold:.2f}\n\n"
        results_text += "Try lowering the confidence threshold or uploading a different image."
        return annotated_image, results_text, json.dumps({"detections": [], "count": 0})
    
    results_lines.append(f"🎯 Detected {len(boxes)} objects:\n")
    results_lines.append("=" * 60 + "\n")
    
    # Draw boxes and collect results
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        # Skip N/A classes
        if class_name == "N/A":
            continue
        
        x1, y1, x2, y2 = box
        color = tuple(int(c) for c in COLORS[label % len(COLORS)])
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Prepare label text
        label_text = f"{class_name}"
        conf_text = f"{score:.1%}"
        
        # Draw label background
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Background for label
        draw.rectangle(
            [x1, y1 - text_height - 8, x1 + text_width + 10, y1],
            fill=color
        )
        draw.text((x1 + 5, y1 - text_height - 4), label_text, fill='white', font=font)
        
        # Draw confidence on the box
        draw.text((x1 + 5, y1 + 5), conf_text, fill=color, font=small_font)
        
        # Add to results
        detection_info = {
            "object": class_name,
            "confidence": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        }
        detections.append(detection_info)
        
        # Format text results
        results_lines.append(f"📦 {idx + 1}. {class_name.upper()}\n")
        results_lines.append(f"   Confidence: {score:.1%} ({score:.4f})\n")
        results_lines.append(f"   Location: [{int(x1)}, {int(y1)}] → [{int(x2)}, {int(y2)}]\n")
        results_lines.append(f"   Center: ({int((x1+x2)/2)}, {int((y1+y2)/2)})\n")
        results_lines.append(f"   Size: {int(x2-x1)} × {int(y2-y1)} pixels\n")
        results_lines.append("-" * 60 + "\n")
    
    results_text = "".join(results_lines)
    
    # Create JSON output
    results_json_str = json.dumps({
        "total_detections": len(detections),
        "confidence_threshold": float(confidence_threshold),
        "image_size": [int(image.width), int(image.height)],
        "detections": detections
    }, indent=2)
    
    return annotated_image, results_text, results_json_str


def save_feedback(image, predictions_json, is_correct, corrections_text, notes):
    """
    Save user feedback to database
    
    Args:
        image: PIL Image
        predictions_json: JSON string of predictions
        is_correct: Boolean from user confirmation
        corrections_text: User's corrections as text
        notes: Additional notes from user
    """
    if image is None:
        return "⚠️ No image to save feedback for."
    
    try:
        # Parse predictions
        predictions = json.loads(predictions_json) if predictions_json else {}
        predicted_objects = predictions.get('detections', [])
        confidence = predictions.get('confidence_threshold', 0.5)
        
        # Parse corrections
        corrections = None
        if corrections_text and corrections_text.strip():
            corrections = {"user_input": corrections_text}
        
        # Save to database
        feedback_id = feedback_db.save_feedback(
            image=image,
            predicted_objects=predicted_objects,
            user_confirmed=is_correct,
            user_corrections=corrections,
            confidence_threshold=confidence,
            notes=notes or ""
        )
        
        # Get stats
        stats = feedback_db.get_feedback_stats()
        
        return f"""✅ **Feedback Saved Successfully!**

Feedback ID: {feedback_id}
Total Feedback Collected: {stats['total']}
✓ Confirmed: {stats['confirmed']}
✗ Corrected: {stats['corrected']}

Thank you for helping improve the model!"""
        
    except Exception as e:
        return f"❌ Error saving feedback: {str(e)}"


def get_feedback_statistics():
    """Get current feedback statistics"""
    try:
        stats = feedback_db.get_feedback_stats()
        
        accuracy = 0
        if stats['total'] > 0:
            accuracy = (stats['confirmed'] / stats['total']) * 100
        
        return f"""📊 **Feedback Statistics**

**Total Submissions:** {stats['total']}
**Correct Predictions:** {stats['confirmed']} ({accuracy:.1f}%)
**Needed Corrections:** {stats['corrected']}

💡 Your feedback helps improve the model!"""
    except Exception as e:
        return f"Error: {str(e)}"


def create_ui():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Object Detection AI") as app:
        
        gr.Markdown("""
        # 🤖 Object Detection AI - Maximum Sensitivity
        ### Detect 80 Object Categories (Trained on 330,000+ Images)
        
        Upload an image and the AI will identify and locate all objects with bounding boxes.
        **Powered by Faster R-CNN** trained on the massive COCO dataset (1.5 million+ labeled instances).
        
        ⚠️ **Note:** Can only detect objects from the 80 COCO categories (see list below).
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.99,
                    value=0.2,
                    step=0.05,
                    label="🎯 Confidence Threshold",
                    info="Lower = more detections (may include false positives). Start at 0.2 for maximum sensitivity."
                )
                
                detect_btn = gr.Button(
                    "🔍 Detect Objects",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 📋 Detectable Categories
                
                **People & Animals:** person, cat, dog, horse, bird, cow, elephant, bear, sheep, zebra, giraffe
                
                **Vehicles:** car, bicycle, motorcycle, bus, truck, train, airplane, boat
                
                **Indoor:** chair, couch, bed, table, tv, laptop, keyboard, mouse, book, clock, bottle, cup, bowl
                
                **Food:** banana, apple, pizza, cake, sandwich, hot dog, donut, orange, broccoli, carrot
                
                **And 50+ more categories!**
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### ✨ Detection Results")
                output_image = gr.Image(
                    label="Detected Objects",
                    type="pil",
                    height=400
                )
                
                with gr.Tabs():
                    with gr.Tab("📝 Detailed Results"):
                        results_text = gr.Textbox(
                            label="Detection Details",
                            lines=15,
                            max_lines=20
                        )
                    
                    with gr.Tab("💾 JSON Output"):
                        results_json = gr.Textbox(
                            label="JSON Data",
                            lines=15
                        )
                
                # Feedback Section
                gr.Markdown("---")
                gr.Markdown("### 📝 Feedback: Help Improve the Model")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        **Is the detection correct?**  
                        Your feedback helps improve accuracy!
                        """)
                        
                        with gr.Row():
                            confirm_correct = gr.Button("✅ Yes, Correct", variant="primary", size="sm")
                            confirm_wrong = gr.Button("❌ No, Wrong", variant="stop", size="sm")
                        
                        corrections_input = gr.Textbox(
                            label="What should it be? (Optional)",
                            placeholder="e.g., 'Should detect 2 dogs and 1 cat'",
                            lines=2,
                            visible=False
                        )
                        
                        notes_input = gr.Textbox(
                            label="Additional Notes (Optional)",
                            placeholder="Any comments about the detection quality...",
                            lines=2
                        )
                        
                        submit_feedback_btn = gr.Button("💾 Submit Feedback", variant="secondary", visible=False)
                        
                    with gr.Column(scale=1):
                        feedback_status = gr.Markdown("""""")
                        
                        stats_display = gr.Markdown(get_feedback_statistics())
                        
                        refresh_stats_btn = gr.Button("🔄 Refresh Statistics", size="sm")
        
        # Examples
        gr.Markdown("### 🖼️ Try These Examples")
        gr.Examples(
            examples=[
                ["examples/street.jpg", 0.5],
                ["examples/kitchen.jpg", 0.5],
                ["examples/pets.jpg", 0.5],
            ],
            inputs=[input_image, confidence_slider],
            label="Click to load example images"
        )
        
        # Event handler for detection
        detect_btn.click(
            fn=predict_objects,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, results_text, results_json]
        )
        
        # Also detect on image upload
        input_image.change(
            fn=predict_objects,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, results_text, results_json]
        )
        
        # Feedback event handlers
        def show_correct_feedback():
            return {
                corrections_input: gr.update(visible=False),
                submit_feedback_btn: gr.update(visible=True),
                feedback_status: "✅ **Marked as Correct** - Click 'Submit Feedback' to save"
            }
        
        def show_wrong_feedback():
            return {
                corrections_input: gr.update(visible=True),
                submit_feedback_btn: gr.update(visible=True),
                feedback_status: "❌ **Marked as Wrong** - Please tell us what's correct, then submit"
            }
        
        confirm_correct.click(
            fn=show_correct_feedback,
            outputs=[corrections_input, submit_feedback_btn, feedback_status]
        )
        
        confirm_wrong.click(
            fn=show_wrong_feedback,
            outputs=[corrections_input, submit_feedback_btn, feedback_status]
        )
        
        # Save feedback when "Yes, Correct" is clicked and submitted
        def submit_correct_feedback(image, predictions, notes):
            result = save_feedback(image, predictions, True, "", notes)
            return {
                feedback_status: result,
                submit_feedback_btn: gr.update(visible=False),
                corrections_input: gr.update(visible=False, value=""),
                notes_input: gr.update(value=""),
                stats_display: get_feedback_statistics()
            }
        
        # Save feedback when "No, Wrong" is clicked and submitted  
        def submit_wrong_feedback(image, predictions, corrections, notes):
            result = save_feedback(image, predictions, False, corrections, notes)
            return {
                feedback_status: result,
                submit_feedback_btn: gr.update(visible=False),
                corrections_input: gr.update(visible=False, value=""),
                notes_input: gr.update(value=""),
                stats_display: get_feedback_statistics()
            }
        
        # Determine which feedback to submit based on whether corrections are provided
        def submit_feedback_handler(image, predictions, corrections, notes):
            if corrections and corrections.strip():
                return submit_wrong_feedback(image, predictions, corrections, notes)
            else:
                return submit_correct_feedback(image, predictions, notes)
        
        submit_feedback_btn.click(
            fn=submit_feedback_handler,
            inputs=[input_image, results_json, corrections_input, notes_input],
            outputs=[feedback_status, submit_feedback_btn, corrections_input, notes_input, stats_display]
        )
        
        # Refresh statistics
        refresh_stats_btn.click(
            fn=get_feedback_statistics,
            outputs=[stats_display]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About
        
        **Model:** Faster R-CNN with ResNet-50 backbone  
        **Training Data:** COCO Dataset (330K+ images, 1.5M+ instances)  
        **Categories:** 80 object classes  
        **Device:** """ + str(device) + """
        
        **How to use:**
        1. Upload an image or select an example
        2. Adjust the confidence threshold (default: 0.5)
        3. View detected objects with bounding boxes
        4. Check detailed results or export as JSON
        
        **Tips:**
        - Lower threshold → More detections (may include false positives)
        - Higher threshold → Fewer, more confident detections
        - Works best with clear, well-lit images
        - Supports multiple objects in one image
        """)
    
    return app


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("OBJECT DETECTION WEB APPLICATION")
    print("=" * 70)
    
    # Initialize model
    print("\n🔧 Initializing model...")
    initialize_model()
    
    # Create and launch UI
    print("\n🚀 Launching web interface...")
    app = create_ui()
    
    # Launch with public link option
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )
