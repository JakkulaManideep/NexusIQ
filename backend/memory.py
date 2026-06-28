import os
import json
from datetime import datetime

# Get path to strategic.json relative to memory.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIC_MEM_PATH = os.path.join(BASE_DIR, "data", "strategic.json")

def save_feedback(customer_id: str, recommendation: str, action: str) -> dict:
    """
    Saves a CSM feedback action (Accept, Modify, Reject) for a recommendation to strategic.json.
    """
    os.makedirs(os.path.dirname(STRATEGIC_MEM_PATH), exist_ok=True)
    
    # Load existing feedback
    feedback_list = []
    if os.path.exists(STRATEGIC_MEM_PATH):
        try:
            with open(STRATEGIC_MEM_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    feedback_list = json.loads(content)
        except Exception as e:
            print(f"Error reading strategic memory: {e}")
            feedback_list = []

    # Create new entry
    new_entry = {
        "customer_id": customer_id,
        "recommendation": recommendation,
        "action": action,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    feedback_list.append(new_entry)

    # Save back to file
    try:
        with open(STRATEGIC_MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=2)
    except Exception as e:
        print(f"Error writing to strategic memory: {e}")
        return {"error": str(e)}

    return new_entry

def get_last_feedback(customer_id: str) -> dict:
    """
    Retrieves the most recent feedback entry recorded for a specific customer.
    """
    if not os.path.exists(STRATEGIC_MEM_PATH):
        return None

    try:
        with open(STRATEGIC_MEM_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return None
            feedback_list = json.loads(content)
    except Exception as e:
        print(f"Error reading strategic memory: {e}")
        return None

    # Filter feedback for this customer
    customer_feedback = [entry for entry in feedback_list if entry.get("customer_id") == customer_id]
    
    if not customer_feedback:
        return None

    # Sort by timestamp (since datetime format sorts lexicographically, it works natively)
    customer_feedback.sort(key=lambda x: x.get("timestamp", ""))
    
    # Return last entry
    return customer_feedback[-1]
