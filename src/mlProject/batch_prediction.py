from flask import Blueprint, request, jsonify
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

batch_bp = Blueprint('batch', __name__)

@batch_bp.route('/predict/batch', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Invalid or missing JSON body (expected Content-Type: application/json)'}), 400
        if 'instances' not in data:
            return jsonify({'error': 'Missing instances array in request'}), 400
            
        df = pd.DataFrame(data['instances'])
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(df)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
