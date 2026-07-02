"""
Phase 26: Enterprise Cost Optimization Engine
Tracks infrastructure spending and recommends optimization opportunities.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

COST_FILE = Path("artifacts/cost_optimizer.json")


class EnterpriseCostOptimizer:
    """Analyzes ML infrastructure costs and generates savings recommendations."""

    def __init__(self):
        self._ensure_data()

    def _ensure_data(self):
        COST_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not COST_FILE.exists():
            now = datetime.utcnow()
            data = {
                "monthly_budgets": {"AWS": 1500, "Azure": 1200, "GCP": 600, "On-Premises": 0},
                "actual_spend": {
                    "AWS": 1240.50, "Azure": 890.25, "GCP": 430.75, "On-Premises": 0
                },
                "cost_breakdown": {
                    "compute": 1450.00, "storage": 320.50, "networking": 180.00,
                    "licensing": 250.00, "monitoring": 171.00
                },
                "weekly_trend": [
                    {"week": "W1", "cost": 560.20},
                    {"week": "W2", "cost": 612.80},
                    {"week": "W3", "cost": 590.45},
                    {"week": "W4", "cost": 608.05}
                ]
            }
            COST_FILE.write_text(json.dumps(data, indent=2))

    def _load(self):
        return json.loads(COST_FILE.read_text())

    def get_cost_report(self) -> dict:
        data = self._load()
        total_budget = sum(data["monthly_budgets"].values())
        total_spend = sum(data["actual_spend"].values())
        utilization_pct = round((total_spend / total_budget) * 100, 1) if total_budget else 0
        return {
            "report_period": "Current Month",
            "report_generated_at": datetime.utcnow().isoformat() + "Z",
            "total_budget_usd": total_budget,
            "total_spend_usd": round(total_spend, 2),
            "budget_utilization_pct": utilization_pct,
            "by_provider": {
                provider: {
                    "budget": data["monthly_budgets"][provider],
                    "spend": data["actual_spend"][provider],
                    "utilization_pct": round(
                        (data["actual_spend"][provider] / data["monthly_budgets"][provider] * 100), 1
                    ) if data["monthly_budgets"][provider] > 0 else 0
                }
                for provider in data["monthly_budgets"]
            },
            "cost_breakdown": data["cost_breakdown"],
            "weekly_trend": data["weekly_trend"]
        }

    def get_forecast(self) -> dict:
        data = self._load()
        total_spend = sum(data["actual_spend"].values())
        # Simple linear extrapolation
        trend = data["weekly_trend"]
        avg_weekly = sum(t["cost"] for t in trend) / len(trend) if trend else 600
        forecast_30d = avg_weekly * 4
        return {
            "forecast_period": "Next 30 Days",
            "forecast_generated_at": datetime.utcnow().isoformat() + "Z",
            "projected_total_usd": round(forecast_30d, 2),
            "projected_vs_budget_pct": round((forecast_30d / sum(data["monthly_budgets"].values())) * 100, 1),
            "confidence": "medium",
            "weekly_projections": [
                {"week": f"W{i+1}", "projected_cost": round(avg_weekly * (1 + 0.02 * i), 2)}
                for i in range(4)
            ]
        }

    def get_recommendations(self) -> dict:
        recommendations = [
            {
                "category": "Compute Right-sizing",
                "priority": "HIGH",
                "description": "3 SageMaker endpoints running at <20% CPU — downgrade instance type",
                "estimated_savings_usd_month": 340.00,
                "effort": "Low (1-2 hours)"
            },
            {
                "category": "Storage Tiering",
                "priority": "MEDIUM",
                "description": "Move model artifacts older than 90 days to S3 Glacier",
                "estimated_savings_usd_month": 85.50,
                "effort": "Low (automated lifecycle policy)"
            },
            {
                "category": "Reserved Instances",
                "priority": "MEDIUM",
                "description": "Convert 2 on-demand training VMs to 1-year reserved — 40% discount",
                "estimated_savings_usd_month": 210.00,
                "effort": "Medium (procurement approval needed)"
            },
            {
                "category": "Spot/Preemptible Instances",
                "priority": "LOW",
                "description": "Use GCP preemptible VMs for batch retraining jobs",
                "estimated_savings_usd_month": 130.00,
                "effort": "Medium (job needs checkpoint support)"
            }
        ]
        total_savings = sum(r["estimated_savings_usd_month"] for r in recommendations)
        return {
            "recommendations": recommendations,
            "total_estimated_savings_usd_month": total_savings,
            "payback_period_days": 30,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
