import os
from datetime import timedelta

# Fallback only to secure random in prod, never hardcode
JWT_SECRET = os.environ.get("JWT_SECRET_KEY")
if not JWT_SECRET:
    import secrets
    JWT_SECRET = secrets.token_urlsafe(32)

JWT_EXPIRATION_DELTA = timedelta(hours=1)

def verify_token(token):
    # JWT verification logic would go here
    pass