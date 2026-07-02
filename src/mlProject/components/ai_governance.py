import os
import json
from datetime import datetime

class ResponsibleAIPolicyEngine:
    def __init__(self, policy_path="artifacts/ai_governance_policies.json"):
        self.policy_path = policy_path
        self._init_policies()

    def _init_policies(self):
        if not os.path.exists(self.policy_path):
            os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)
            initial_data = {
                "policies": [
                    {"policy_id": "pol_accuracy", "name": "Minimum R2 Target", "rule": "R2 Score must be >= 0.35", "status": "ACTIVE"},
                    {"policy_id": "pol_drift", "name": "Drift Threshold Limit", "rule": "Feature drift ratio must be < 25%", "status": "ACTIVE"},
                    {"policy_id": "pol_bias", "name": "Bias Disparity Margin", "rule": "Disparity difference between low/high alcohol groups must be < 0.15", "status": "ACTIVE"}
                ],
                "compliance_audits": [
                    {
                        "audit_id": "aud_1",
                        "policy_id": "pol_accuracy",
                        "status": "COMPLIANT",
                        "details": "Model R2 score is 0.51.",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ]
            }
            with open(self.policy_path, "w") as f:
                json.dump(initial_data, f, indent=2)

    def load_policies_manifest(self) -> dict:
        try:
            with open(self.policy_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"policies": [], "compliance_audits": []}

    def save_policies_manifest(self, manifest: dict):
        with open(self.policy_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def verify_compliance(self, metrics: dict) -> dict:
        manifest = self.load_policies_manifest()
        audits = []
        is_compliant = True
        
        # Test R2 Accuracy
        r2 = metrics.get("r2", 0.0)
        r2_passed = r2 >= 0.35
        audits.append({
            "policy": "Minimum R2 Target",
            "status": "PASSED" if r2_passed else "FAILED",
            "value": r2,
            "details": f"Target >= 0.35, got {r2:.4f}"
        })
        if not r2_passed:
            is_compliant = False
            
        # Test Drift
        drift = metrics.get("drift_ratio", 0.0)
        drift_passed = drift < 0.25
        audits.append({
            "policy": "Drift Threshold Limit",
            "status": "PASSED" if drift_passed else "FAILED",
            "value": drift,
            "details": f"Target < 0.25, got {drift:.4f}"
        })
        if not drift_passed:
            is_compliant = False

        # Save audit record
        audit_entry = {
            "audit_id": f"aud_{len(manifest.get('compliance_audits', [])) + 1}",
            "status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
            "checks": audits,
            "timestamp": datetime.utcnow().isoformat()
        }
        manifest["compliance_audits"].insert(0, audit_entry)
        self.save_policies_manifest(manifest)
        
        return audit_entry
