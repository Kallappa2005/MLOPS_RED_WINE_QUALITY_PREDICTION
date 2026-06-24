import threading

_lock = threading.Lock()
retraining_in_progress = False

def start_retraining():
    global retraining_in_progress
    with _lock:
        if retraining_in_progress:
            return False
        retraining_in_progress = True
    
    try:
        print("Retraining model...")
        # Simulate retraining logic here
    finally:
        with _lock:
            retraining_in_progress = False
    return True