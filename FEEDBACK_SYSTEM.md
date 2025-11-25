# 📝 User Feedback System

## Overview

The Object Detection AI now includes a comprehensive feedback system that allows users to confirm or correct predictions, helping improve model accuracy over time.

## Features

### ✅ **Feedback Collection**
- **Confirm Correct**: Mark predictions as accurate
- **Report Wrong**: Indicate incorrect predictions and provide corrections
- **Add Notes**: Include additional context or comments
- **Real-time Statistics**: Track accuracy and feedback trends

### 💾 **Database Storage**
- All feedback is automatically saved to `feedback.db`
- Stores predictions, user confirmations, corrections, and metadata
- Tracks image hashes to identify repeat images
- Includes timestamps for trend analysis

## How to Use

### In the Web Interface

1. **Upload an image** and wait for detection results

2. **Review the predictions** in the "Detection Results" section

3. **Provide Feedback**:
   - Click **"✅ Yes, Correct"** if predictions are accurate
   - Click **"❌ No, Wrong"** if predictions need correction
   
4. **If Wrong**:
   - Enter what the correct detections should be
   - Example: "Should detect 2 dogs and 1 cat"
   
5. **Add Notes** (optional):
   - Provide context like "Image was blurry" or "Object partially hidden"

6. **Click "💾 Submit Feedback"** to save

7. **View Statistics** to see overall accuracy

### View Collected Feedback

```bash
# Activate virtual environment
source venv/bin/activate

# View all feedback in terminal
python view_feedback.py

# Export to CSV
python view_feedback.py --export
```

## Database Schema

### `feedback` Table
- `id`: Unique feedback ID
- `image_hash`: MD5 hash of the image
- `timestamp`: When feedback was submitted
- `predicted_objects`: JSON array of predictions
- `user_confirmed`: Boolean (correct/wrong)
- `user_corrections`: User's correction text
- `confidence_threshold`: Threshold used for detection
- `notes`: Additional user comments

### `image_cache` Table
- `image_hash`: MD5 hash (primary key)
- `image_size`: Width x Height
- `first_seen`: First time this image was processed

## Use Cases

### 1. **Model Improvement**
- Identify common failure patterns
- Understand which objects are frequently misdetected
- Collect training data for fine-tuning

### 2. **Performance Monitoring**
- Track accuracy over time
- Compare performance across different confidence thresholds
- Identify problematic image types

### 3. **User Engagement**
- Allow users to contribute to model improvement
- Build trust through transparency
- Create feedback loop for continuous improvement

## Statistics Dashboard

The interface shows real-time statistics:
- **Total Submissions**: All feedback entries
- **Correct Predictions**: Confirmed accurate detections
- **Needed Corrections**: Flagged incorrect predictions
- **Accuracy Percentage**: Overall prediction accuracy

## Data Export

Export feedback for analysis:

```python
from feedback_db import feedback_db

# Export to CSV
feedback_db.export_feedback_csv("my_export.csv")

# Get statistics
stats = feedback_db.get_feedback_stats()
print(f"Accuracy: {(stats['confirmed'] / stats['total']) * 100:.1f}%")

# Get all records
records = feedback_db.get_all_feedback()
```

## Privacy & Storage

- **Local Storage**: All data stored locally in `feedback.db`
- **No External Upload**: Feedback never leaves your machine
- **Image Hashing**: Images stored as hashes, not actual files
- **SQLite Database**: Lightweight, portable, no server required

## Future Enhancements

Potential improvements:
- [ ] Active learning: Retrain on corrected data
- [ ] Export feedback for human annotation tools
- [ ] Integration with model retraining pipeline
- [ ] Visualization dashboard for feedback trends
- [ ] Multi-user feedback aggregation
- [ ] Confidence calibration based on feedback

## Troubleshooting

**Feedback not saving?**
- Check if `feedback.db` exists in the project directory
- Verify write permissions
- Check console for error messages

**Want to reset feedback?**
```bash
# Delete the database file
rm feedback.db

# Or use Python
from feedback_db import feedback_db
import os
os.remove(feedback_db.db_path)
```

**View database directly:**
```bash
sqlite3 feedback.db
sqlite> SELECT COUNT(*) FROM feedback;
sqlite> .quit
```

## Benefits

✅ **Continuous Improvement**: Build better models with user input
✅ **Quality Assurance**: Track real-world performance
✅ **User Empowerment**: Let users contribute meaningfully
✅ **Data Collection**: Gather labeled data for future training
✅ **Transparency**: Show users their impact through statistics

---

**Start collecting feedback now!** Every submission helps improve the model's accuracy. 🚀
