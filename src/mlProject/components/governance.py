import os
import json
from datetime import datetime

class ModelGovernanceEngine:
    def __init__(self, governance_path="artifacts/governance_manifest.json"):
        self.governance_path = governance_path
        self._init_manifest()

    def _init_manifest(self):
        if not os.path.exists(self.governance_path):
            os.makedirs(os.path.dirname(self.governance_path), exist_ok=True)
            initial_data = {
                "approvals": [],
                "compliance_checks": [],
                "audit_history": []
            }
            with open(self.governance_path, "w") as f:
                json.dump(initial_data, f, indent=2)

    def request_promotion_approval(self, version_id: str, requested_by: str) -> dict:
        """
        Record a promotion request that remains pending until Admin review.
        """
        manifest = self.load_manifest()
        request_info = {
            "version_id": version_id,
            "requested_by": requested_by,
            "status": "PENDING_APPROVAL",
            "timestamp": datetime.utcnow().isoformat(),
            "approved_by": None
        }
        manifest["approvals"].insert(0, request_info)
        self.save_manifest(manifest)
        return request_info

    def approve_promotion(self, version_id: str, approved_by: str) -> bool:
        manifest = self.load_manifest()
        updated = False
        for app in manifest.get("approvals", []):
            if app.get("version_id") == version_id and app.get("status") == "PENDING_APPROVAL":
                app["status"] = "APPROVED"
                app["approved_by"] = approved_by
                app["approved_at"] = datetime.utcnow().isoformat()
                updated = True
                
        if updated:
            manifest["audit_history"].insert(0, {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "PROMOTION_APPROVED",
                "user": approved_by,
                "version_id": version_id
            })
            self.save_manifest(manifest)
        return updated

    def run_compliance_check(self, version_id: str, rmse: float, r2: float) -> dict:
        """
        Scan a model version against legal/business thresholds.
        """
        manifest = self.load_manifest()
        
        # Business requirements: RMSE must be <= 0.65, R2 must be >= 0.35
        rmse_passed = rmse <= 0.65
        r2_passed = r2 >= 0.35
        overall_passed = rmse_passed and r2_passed
        
        check_result = {
            "version_id": version_id,
            "timestamp": datetime.utcnow().isoformat(),
            "rules": {
                "rmse_limit_0.65": {"value": rmse, "status": "PASSED" if rmse_passed else "FAILED"},
                "r2_limit_0.35": {"value": r2, "status": "PASSED" if r2_passed else "FAILED"}
            },
            "status": "COMPLIANT" if overall_passed else "NON_COMPLIANT"
        }
        
        manifest["compliance_checks"].insert(0, check_result)
        self.save_manifest(manifest)
        return check_result

    def load_manifest(self) -> dict:
        try:
            with open(self.governance_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"approvals": [], "compliance_checks": [], "audit_history": []}

    def save_manifest(self, manifest: dict):
        with open(self.governance_path, "w") as f:
            json.dump(manifest, f, indent=2)
