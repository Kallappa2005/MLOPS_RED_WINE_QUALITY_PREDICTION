import os
import json
from datetime import datetime

class MLOpsAgentCoordinator:
    def __init__(self, logs_path="artifacts/agents_logs.json"):
        self.logs_path = logs_path
        self._init_logs()

    def _init_logs(self):
        if not os.path.exists(self.logs_path):
            os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)
            initial_data = [
                {
                    "agent": "MonitoringAgent",
                    "task": "System resource scans",
                    "status": "COMPLETED",
                    "details": "All system hardware resources are within stable parameters.",
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "agent": "GovernanceAgent",
                    "task": "Model bias verification checks",
                    "status": "COMPLETED",
                    "details": "Model bias checks passed successfully.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
            with open(self.logs_path, "w") as f:
                json.dump(initial_data, f, indent=2)

    def get_agent_activities(self) -> list:
        try:
            with open(self.logs_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def record_activity(self, agent_name: str, task: str, status: str, details: str) -> bool:
        try:
            activities = self.get_agent_activities()
            activities.insert(0, {
                "agent": agent_name,
                "task": task,
                "status": status,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            })
            with open(self.logs_path, "w") as f:
                json.dump(activities, f, indent=2)
            return True
        except Exception:
            return False
            
    def run_agent_loop(self) -> list:
        """
        Simulate autonomous agent execution loop across all active tasks.
        """
        self.record_activity("MonitoringAgent", "Data drift validation", "COMPLETED", "Feature drift check: No significant deviations.")
        self.record_activity("TrainingAgent", "Auto retraining check", "COMPLETED", "Baseline R2 score stable. Retraining is not triggered.")
        self.record_activity("OptimizationAgent", "Hyperparameter sweep check", "COMPLETED", "Simulated tuning trials ran: Best R2: 0.52.")
        return self.get_agent_activities()
