import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class FeedbackHandler:
    def __init__(self, feedback_file='feedback.json'):
        self.feedback_file = feedback_file
        self._ensure_feedback_file()

    def _ensure_feedback_file(self):
        """Ensure the feedback file exists and is properly initialized."""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)
            logging.info(f"Created new feedback file: {self.feedback_file}")

    def save_feedback(self, email, rating, feedback_type, message):
        """Save feedback to the JSON file."""
        try:
            # Load existing feedback
            feedback_list = self.load_feedback()
            
            # Create new feedback entry
            new_feedback = {
                "id": len(feedback_list) + 1,
                "email": email,
                "rating": rating,
                "type": feedback_type,
                "message": message,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add new feedback to list
            feedback_list.append(new_feedback)
            
            # Save updated feedback list
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_list, f, indent=4)
            
            logging.info(f"Saved new feedback with ID: {new_feedback['id']}")
            return True
        except Exception as e:
            logging.error(f"Error saving feedback: {str(e)}")
            return False

    def load_feedback(self):
        """Load all feedback from the JSON file."""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Error loading feedback: {str(e)}")
            return []

    def get_feedback_stats(self):
        """Get statistics about the feedback."""
        feedback_list = self.load_feedback()
        if not feedback_list:
            return {
                "total": 0,
                "average_rating": 0,
                "type_distribution": {},
                "recent_feedback": []
            }

        # Calculate statistics
        total = len(feedback_list)
        ratings = [int(f['rating']) for f in feedback_list if f['rating'].isdigit()]
        average_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Count feedback types
        type_distribution = {}
        for feedback in feedback_list:
            feedback_type = feedback['type']
            type_distribution[feedback_type] = type_distribution.get(feedback_type, 0) + 1

        # Get recent feedback (last 5)
        recent_feedback = sorted(feedback_list, key=lambda x: x['date'], reverse=True)[:5]

        return {
            "total": total,
            "average_rating": round(average_rating, 2),
            "type_distribution": type_distribution,
            "recent_feedback": recent_feedback
        } 