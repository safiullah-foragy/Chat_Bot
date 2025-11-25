"""
View Feedback Data
Quick script to view collected feedback
"""

from feedback_db import feedback_db
import json
from datetime import datetime

def view_feedback():
    """Display all feedback in a readable format"""
    
    print("=" * 80)
    print("OBJECT DETECTION FEEDBACK DATABASE")
    print("=" * 80)
    
    # Get statistics
    stats = feedback_db.get_feedback_stats()
    
    print(f"\n📊 STATISTICS")
    print(f"   Total Feedback: {stats['total']}")
    print(f"   ✅ Confirmed Correct: {stats['confirmed']}")
    print(f"   ❌ Needs Correction: {stats['corrected']}")
    
    if stats['total'] > 0:
        accuracy = (stats['confirmed'] / stats['total']) * 100
        print(f"   🎯 Accuracy: {accuracy:.1f}%")
    
    print(f"\n" + "=" * 80)
    
    # Get all feedback
    records = feedback_db.get_all_feedback()
    
    if not records:
        print("\n⚠️  No feedback data yet. Start using the app and submit feedback!")
        return
    
    print(f"\n📝 FEEDBACK RECORDS ({len(records)} total)\n")
    
    for i, record in enumerate(records[:20], 1):  # Show last 20
        record_id, img_hash, timestamp, predicted, confirmed, corrections, threshold, notes = record
        
        print(f"\n{'─' * 80}")
        print(f"Record #{record_id} | {timestamp}")
        print(f"{'─' * 80}")
        print(f"Image Hash: {img_hash[:16]}...")
        print(f"Confidence Threshold: {threshold}")
        print(f"User Confirmed: {'✅ YES' if confirmed else '❌ NO'}")
        
        # Parse predictions
        if predicted:
            preds = json.loads(predicted)
            print(f"\nPredicted Objects ({len(preds)}):")
            for j, obj in enumerate(preds[:5], 1):  # Show first 5
                print(f"   {j}. {obj.get('object', 'N/A')} ({obj.get('confidence', 0):.2%})")
            if len(preds) > 5:
                print(f"   ... and {len(preds) - 5} more")
        
        # Show corrections if any
        if corrections:
            corr = json.loads(corrections)
            print(f"\n❌ User Corrections:")
            print(f"   {corr.get('user_input', 'N/A')}")
        
        # Show notes
        if notes:
            print(f"\n💬 Notes: {notes}")
    
    if len(records) > 20:
        print(f"\n... and {len(records) - 20} more records")
    
    print(f"\n" + "=" * 80)
    print(f"💾 Database location: {feedback_db.db_path}")
    print(f"=" * 80)


def export_feedback():
    """Export feedback to CSV"""
    output_file = feedback_db.export_feedback_csv()
    print(f"\n✅ Feedback exported to: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_feedback()
    else:
        view_feedback()
        
        print("\n💡 Tip: Run 'python view_feedback.py --export' to export to CSV")
