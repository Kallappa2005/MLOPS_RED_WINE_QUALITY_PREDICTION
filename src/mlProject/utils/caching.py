import functools
import hashlib
import json

def cache_prediction(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a simple hash of the arguments
        key = hashlib.md5(json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True).encode('utf-8')).hexdigest()
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper
