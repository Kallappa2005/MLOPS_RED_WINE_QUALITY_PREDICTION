import mlflow
from mlflow.entities import ViewType
from datetime import datetime
from mlProject import logger
from mlProject.config.configuration import ConfigurationManager

def get_mlflow_runs():
    try:
        config_manager = ConfigurationManager()
        registry_config = config_manager.get_model_registry_config()
        
        mlflow.set_tracking_uri(registry_config.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        experiment = client.get_experiment_by_name(registry_config.mlflow_experiment_name)
        if experiment is None:
            return {"enabled": True, "runs": []}
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"]
        )
        
        runs_list = []
        for run in runs:
            metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
            runs_list.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or run.data.tags.get("mlflow.runName", "Unnamed"),
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                "metrics": metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
        return {"enabled": True, "runs": runs_list}
    except Exception as e:
        logger.error(f"Failed to fetch mlflow runs: {e}")
        return {
            "enabled": True,
            "error": str(e),
            "runs": [
                {
                    "run_id": "mock_elasticnet_run_01",
                    "run_name": "elastic_net_baseline",
                    "status": "FINISHED",
                    "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "metrics": {"r2": 0.3552, "rmse": 0.6469, "mae": 0.5063},
                    "params": {"alpha": "0.2", "l1_ratio": "0.1"},
                    "tags": {"model_type": "ElasticNet"}
                }
            ]
        }
