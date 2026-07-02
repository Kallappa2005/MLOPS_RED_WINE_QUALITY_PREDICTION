import jwt
from datetime import datetime, timedelta
import functools
from flask import request, jsonify
import sqlite3
import os

import secrets
from werkzeug.security import generate_password_hash, check_password_hash

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
JWT_ALGORITHM = "HS256"

# Passwords should be injected via environment variables or securely stored.
# Here we use secure hashes, defaulting to env vars if provided.
USER_DB = {
    "admin": {"password_hash": generate_password_hash(os.environ.get("ADMIN_PASSWORD", secrets.token_urlsafe(16))), "role": "Admin"},
    "engineer": {"password_hash": generate_password_hash(os.environ.get("ENGINEER_PASSWORD", secrets.token_urlsafe(16))), "role": "Engineer"},
    "viewer": {"password_hash": generate_password_hash(os.environ.get("VIEWER_PASSWORD", secrets.token_urlsafe(16))), "role": "Viewer"}
}

class AuditLogger:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                username TEXT,
                action TEXT,
                status TEXT,
                ip TEXT,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()
        
    def log_action(self, username, action, status, ip, details=""):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (timestamp, username, action, status, ip, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.utcnow().isoformat(), username, action, status, ip, details))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_logs(self, limit=100):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception:
            return []

def create_token(username, role):
    payload = {
        "exp": datetime.utcnow() + timedelta(hours=4),
        "iat": datetime.utcnow(),
        "sub": username,
        "role": role
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return "Expired"
    except jwt.InvalidTokenError:
        return "Invalid"

def require_role(roles_allowed):
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get("Authorization")
            token = request.args.get("token")
            
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
            if not token:
                AuditLogger().log_action("anonymous", request.path, "DENIED", request.remote_addr, "Missing authentication token")
                return jsonify({"error": "Missing authentication token"}), 401
                
            payload = decode_token(token)
            if payload == "Expired":
                AuditLogger().log_action("anonymous", request.path, "DENIED", request.remote_addr, "Token has expired")
                return jsonify({"error": "Token has expired"}), 401
            elif payload == "Invalid" or not isinstance(payload, dict):
                AuditLogger().log_action("anonymous", request.path, "DENIED", request.remote_addr, "Invalid token")
                return jsonify({"error": "Invalid token"}), 401
                
            user_role = payload.get("role")
            username = payload.get("sub", "unknown")
            
            if user_role not in roles_allowed:
                AuditLogger().log_action(username, request.path, "DENIED", request.remote_addr, f"Role {user_role} not in {roles_allowed}")
                return jsonify({"error": f"Role '{user_role}' is not authorized to access this endpoint"}), 403
                
            AuditLogger().log_action(username, request.path, "GRANTED", request.remote_addr)
            return f(*args, **kwargs)
        return decorated
    return decorator
