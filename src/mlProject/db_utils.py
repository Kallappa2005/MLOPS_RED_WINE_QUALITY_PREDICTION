import sqlite3

def get_logged_predictions(limit=10):
    """Safely fetch logged predictions to prevent SQL injection"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    # Using parameterized query for limit to prevent SQLi
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),))
    results = cursor.fetchall()
    conn.close()
    return results