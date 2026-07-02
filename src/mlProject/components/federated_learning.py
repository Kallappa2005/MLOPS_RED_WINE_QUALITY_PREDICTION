import random
from datetime import datetime

class FederatedCoordinator:
    def __init__(self):
        # Default mock nodes
        self.nodes = ["ClientNode_Alpha", "ClientNode_Beta", "ClientNode_Gamma"]

    def collect_and_aggregate(self) -> dict:
        """
        Simulate collecting model parameters from distributed client nodes and aggregating them.
        """
        node_updates = []
        aggregated_weights = {}

        # Simulating parameter updates (weights represent mock intercepts & slope coefficients)
        features = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]
        
        for node in self.nodes:
            weights = {feat: round(random.uniform(-0.5, 1.5), 4) for feat in features}
            intercept = round(random.uniform(2.5, 4.5), 4)
            node_updates.append({
                "node_id": node,
                "timestamp": datetime.utcnow().isoformat(),
                "samples_count": random.randint(100, 500),
                "weights": weights,
                "intercept": intercept
            })

        # Federated Averaging (FedAvg) implementation simulation
        total_samples = sum(u["samples_count"] for u in node_updates)
        avg_weights = {feat: 0.0 for feat in features}
        avg_intercept = 0.0

        for update in node_updates:
            weight_ratio = update["samples_count"] / total_samples
            avg_intercept += update["intercept"] * weight_ratio
            for feat in features:
                avg_weights[feat] += update["weights"][feat] * weight_ratio

        avg_weights = {k: round(v, 4) for k, v in avg_weights.items()}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "nodes_participated": len(self.nodes),
            "total_samples_aggregated": total_samples,
            "node_details": [
                {"node_id": u["node_id"], "samples": u["samples_count"]} for u in node_updates
            ],
            "aggregated_parameters": {
                "weights": avg_weights,
                "intercept": round(avg_intercept, 4)
            }
        }
