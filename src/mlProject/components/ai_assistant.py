import os
import json
from datetime import datetime

class PredictionAssistant:
    def __init__(self, target_quality=7.0):
        self.target_quality = target_quality

    def explain_prediction_nl(self, features: dict, prediction: float) -> str:
        """
        Generate conversational natural language explanation of the prediction.
        """
        diff = prediction - 5.6  # comparing against average red wine score
        tone = "excellent" if prediction >= 6.5 else "average" if prediction >= 5.0 else "poor"
        
        explanation = (
            f"The model predicted a quality score of {prediction:.2f} ({tone} quality). "
        )
        
        # Explain based on key factors
        alcohol = features.get("alcohol", 10.0)
        volatile_acidity = features.get("volatile acidity", 0.5)
        
        reasons = []
        if alcohol > 11.0:
            reasons.append(f"high alcohol content ({alcohol}%) which correlates strongly with better rating")
        else:
            reasons.append(f"moderate alcohol content ({alcohol}%) which bounds the rating potential")
            
        if volatile_acidity > 0.6:
            reasons.append(f"elevated volatile acidity ({volatile_acidity}), introducing unpleasant sour notes")
        else:
            reasons.append(f"low volatile acidity ({volatile_acidity}), ensuring clean structure")
            
        explanation += "This is primarily influenced by " + " and ".join(reasons) + "."
        return explanation

    def generate_recommendations(self, features: dict, prediction: float) -> list:
        """
        Produce optimization adjustments to push wine score to target_quality.
        """
        recommendations = []
        if prediction >= self.target_quality:
            return ["No changes required! This batch meets or exceeds target standards."]
            
        # Alcohol correction
        alcohol = features.get("alcohol", 10.0)
        if alcohol < 11.5:
            recommendations.append({
                "feature": "alcohol",
                "current": alcohol,
                "recommended": 11.5,
                "action": "Increase fermentation time or sugar addition to raise alcohol volume."
            })
            
        # Volatile acidity correction
        va = features.get("volatile acidity", 0.5)
        if va > 0.4:
            recommendations.append({
                "feature": "volatile acidity",
                "current": va,
                "recommended": 0.35,
                "action": "Improve sulfite management or sanitation checks to limit acetic acid bacteria."
            })
            
        # Sulphates correction
        sulphates = features.get("sulphates", 0.5)
        if sulphates < 0.65:
            recommendations.append({
                "feature": "sulphates",
                "current": sulphates,
                "recommended": 0.70,
                "action": "Increase sulphate dosage slightly to improve preservation and score structure."
            })
            
        return recommendations
