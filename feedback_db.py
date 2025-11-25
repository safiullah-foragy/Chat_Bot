"""
Feedback Database Management
Store user feedback on object detection predictions
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import hashlib


class FeedbackDB:
    def __init__(self, db_path="feedback.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                predicted_objects TEXT,
                user_confirmed BOOLEAN,
                user_corrections TEXT,
                confidence_threshold REAL,
                notes TEXT
            )
        """)
        
        # Create image cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                image_hash TEXT PRIMARY KEY,
                image_size TEXT,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_image_hash(self, image):
        """Generate hash for image"""
        import io
        import numpy as np
        
        if hasattr(image, 'tobytes'):
            img_bytes = image.tobytes()
        else:
            img_array = np.array(image)
            img_bytes = img_array.tobytes()
        
        return hashlib.md5(img_bytes).hexdigest()
    
    def save_feedback(self, image, predicted_objects, user_confirmed, 
                     user_corrections=None, confidence_threshold=0.5, notes=""):
        """
        Save user feedback to database
        
        Args:
            image: PIL Image
            predicted_objects: List of predicted objects
            user_confirmed: Boolean - did user confirm predictions?
            user_corrections: Dict with correct labels
            confidence_threshold: Detection threshold used
            notes: Additional user notes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get image hash
        image_hash = self.get_image_hash(image)
        
        # Store image metadata
        cursor.execute("""
            INSERT OR IGNORE INTO image_cache (image_hash, image_size)
            VALUES (?, ?)
        """, (image_hash, f"{image.width}x{image.height}"))
        
        # Store feedback
        cursor.execute("""
            INSERT INTO feedback 
            (image_hash, predicted_objects, user_confirmed, user_corrections, 
             confidence_threshold, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            image_hash,
            json.dumps(predicted_objects),
            user_confirmed,
            json.dumps(user_corrections) if user_corrections else None,
            confidence_threshold,
            notes
        ))
        
        conn.commit()
        feedback_id = cursor.lastrowid
        conn.close()
        
        return feedback_id
    
    def get_feedback_stats(self):
        """Get statistics about collected feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback count
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]
        
        # Confirmed vs corrected
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_confirmed = 1")
        confirmed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_confirmed = 0")
        corrected = cursor.fetchone()[0]
        
        # Recent feedback
        cursor.execute("""
            SELECT timestamp, predicted_objects, user_confirmed 
            FROM feedback 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent = cursor.fetchall()
        
        conn.close()
        
        return {
            "total": total,
            "confirmed": confirmed,
            "corrected": corrected,
            "recent": recent
        }
    
    def get_all_feedback(self):
        """Get all feedback records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, image_hash, timestamp, predicted_objects, 
                   user_confirmed, user_corrections, confidence_threshold, notes
            FROM feedback
            ORDER BY timestamp DESC
        """)
        
        records = cursor.fetchall()
        conn.close()
        
        return records
    
    def export_feedback_csv(self, output_file="feedback_export.csv"):
        """Export feedback to CSV file"""
        import csv
        
        records = self.get_all_feedback()
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Image Hash", "Timestamp", "Predicted Objects",
                "User Confirmed", "User Corrections", "Confidence Threshold", "Notes"
            ])
            writer.writerows(records)
        
        return output_file


# Global database instance
feedback_db = FeedbackDB()
