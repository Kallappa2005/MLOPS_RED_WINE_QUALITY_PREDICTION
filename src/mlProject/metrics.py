from prometheus_client import Counter, Histogram, generate_latest
from flask import Blueprint, Response

metrics_bp = Blueprint('metrics', __name__)

PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions made')
ERROR_COUNTER = Counter('prediction_errors_total', 'Total number of errors during prediction')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency of predictions')

@metrics_bp.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')
