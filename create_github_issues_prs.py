import sys
import requests

def create_issue(token, repo, title, body):
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"title": title, "body": body}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        issue_url = response.json().get('html_url')
        print(f"Successfully created Issue: {issue_url}")
        return response.json().get("number")
    else:
        print(f"Failed to create Issue '{title}'. Status code: {response.status_code}, Response: {response.text}")
        return None

def create_pr(token, repo, title, body, head, base="main"):
    url = f"https://api.github.com/repos/{repo}/pulls"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"title": title, "body": body, "head": head, "base": base}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print(f"Successfully created Pull Request: {response.json().get('html_url')}")
    else:
        print(f"Failed to create Pull Request '{title}'. Status: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    print("==========================================================")
    print("   GitHub Issues & Pull Requests Automator - RedWine IQ")
    print("==========================================================")
    
    token = input("Enter your GitHub Personal Access Token (PAT): ").strip()
    if not token:
        print("Error: Personal Access Token is required.")
        sys.exit(1)
        
    repo = input("Enter GitHub repository (default: Kallappa2005/MLOPS_RED_WINE_QUALITY_PREDICTION): ").strip()
    if not repo:
        repo = "Kallappa2005/MLOPS_RED_WINE_QUALITY_PREDICTION"
        
    branch = input("Enter branch name containing changes (default: feature/mlops-hub): ").strip()
    if not branch:
        branch = "feature/mlops-hub"
        
    phases = [
        {
            "phase": "Phase 1: Explainable AI (XAI) Dashboard",
            "issue_title": "[FEATURE] Add Explainable AI Dashboard with SHAP-Based Prediction Insights",
            "issue_body": "### Description\nProvide transparency into model predictions using SHAP (SHapley Additive exPlanations) values to explain local predictions and visualize global feature importance.\n\n### Components\n- SHAP Explainer class: src/mlProject/components/xai_explainer.py\n- API: /explain/local and /explain/global\n- UI: Interactive SHAP plots on /dashboard",
            "pr_title": "feat(xai): implement explainable ai dashboard with shap prediction analysis",
            "pr_body": "### Overview\nThis PR integrates SHAP-based local and global predictions insights into the operations dashboard."
        },
        {
            "phase": "Phase 2: Model Monitoring & Drift",
            "issue_title": "[FEATURE] Implement Model Monitoring and Prediction Drift Dashboard",
            "issue_body": "### Description\nEnable prediction logging and distribution drift checks using KS statistics.\n\n### Components\n- Logger: SQLite logs in artifacts/predictions.db\n- Drift Engine: DriftDetector (scipy KS test)\n- API: /monitoring/drift & history",
            "pr_title": "feat(monitoring): implement model monitoring and drift detection platform",
            "pr_body": "### Overview\nThis PR implements real-time monitoring and statistics drift checks using Kolmogorov-Smirnov tests."
        },
        {
            "phase": "Phase 3: MLflow Experiment Tracking",
            "issue_title": "[FEATURE] Add MLflow Experiment Tracking Visualization Dashboard",
            "issue_body": "### Description\nFetch and display experiment training runs dynamically from MLflow server.\n\n### Components\n- Reader: src/mlProject/components/experiment_tracker.py\n- API: /experiments/runs\n- UI: Runs leaderboard table on /dashboard",
            "pr_title": "feat(mlflow): implement experiment tracking and visualization dashboard",
            "pr_body": "### Overview\nThis PR integrates a live model leaderboard and runs history panel directly with MLflow tracking backend."
        },
        {
            "phase": "Phase 4: Multi-Model Benchmarking",
            "issue_title": "[FEATURE] Add Multi-Model Training and Benchmarking Framework",
            "issue_body": "### Description\nExtend training to ElasticNet, RandomForest, GradientBoosting, and XGBoost; auto-select and promote best performing model.\n\n### Components\n- Trainer: Refactored src/mlProject/components/model_trainer.py\n- API: /benchmarking/results\n- UI: Bar chart comparison on /dashboard",
            "pr_title": "feat(training): implement multi-model training and benchmarking system",
            "pr_body": "### Overview\nThis PR upgrades the trainer pipeline to benchmark multiple regressor architectures and register the best one."
        },
        {
            "phase": "Phase 5: Enterprise Prediction Analytics & Exports",
            "issue_title": "[FEATURE] Build Enterprise Prediction Analytics and Reporting Platform",
            "issue_body": "### Description\nAggregate logged predictions daily and support PDF/CSV exports.\n\n### Components\n- Analytics: src/mlProject/components/analytics.py\n- API: /analytics/export/csv and /analytics/export/pdf\n- UI: Volumetric trend line charts",
            "pr_title": "feat(analytics): implement enterprise prediction analytics platform",
            "pr_body": "### Overview\nThis PR implements structured reporting tables and automated PDF reports generation."
        }
    ]

    print("\nStarting issue and PR creation...")
    for p in phases:
        print(f"\n--- Processing {p['phase']} ---")
        issue_num = create_issue(token, repo, p["issue_title"], p["issue_body"])
        if issue_num:
            # Append issue reference to PR description
            pr_desc = f"{p['pr_body']}\n\nCloses #{issue_num}"
            create_pr(token, repo, p["pr_title"], pr_desc, branch)
            
    print("\nAll operations finished.")
