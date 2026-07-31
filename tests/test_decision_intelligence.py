import tempfile
import unittest
from pathlib import Path

from mlProject.components.decision_intelligence import DecisionIntelligenceEngine


class TestDecisionAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "decisions.db")
        self.engine = DecisionIntelligenceEngine(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze_decision_picks_highest_scoring_option(self):
        context = {"cost": -1.0, "quality": 2.0}
        options = [
            {"name": "Vendor A", "cost": 10, "quality": 8},
            {"name": "Vendor B", "cost": 5, "quality": 3},
        ]
        result = self.engine.analyze_decision(context, options)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recommendation"], "Vendor A")
        self.assertIn("confidence", result)
        self.assertIn("reasoning", result)
        self.assertEqual(len(result["ranked_options"]), 2)

    def test_analyze_decision_confidence_is_between_zero_and_one(self):
        context = {"factor": 1.0}
        options = [{"name": "A", "factor": 10}, {"name": "B", "factor": 1}]
        result = self.engine.analyze_decision(context, options)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_analyze_decision_requires_context(self):
        result = self.engine.analyze_decision({}, [{"name": "A", "x": 1}, {"name": "B", "x": 2}])
        self.assertEqual(result["status"], "error")

    def test_analyze_decision_requires_two_options(self):
        result = self.engine.analyze_decision({"x": 1.0}, [{"name": "A", "x": 1}])
        self.assertEqual(result["status"], "error")

    def test_analyze_decision_rejects_option_missing_factor(self):
        context = {"x": 1.0, "y": 1.0}
        options = [{"name": "A", "x": 1}, {"name": "B", "x": 2, "y": 3}]
        result = self.engine.analyze_decision(context, options)
        self.assertEqual(result["status"], "error")

    def test_decision_is_persisted_to_history(self):
        context = {"x": 1.0}
        options = [{"name": "A", "x": 5}, {"name": "B", "x": 1}]
        self.engine.analyze_decision(context, options)
        history = self.engine.get_decision_history()
        self.assertEqual(len(history), 1)

    def test_low_confidence_decision_triggers_alert(self):
        context = {"x": 1.0}
        options = [{"name": "A", "x": 5.0}, {"name": "B", "x": 4.99}]
        self.engine.analyze_decision(context, options)
        alerts = self.engine.get_alerts(status="open")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["alert_type"], "low_confidence_decision")

    def test_high_confidence_decision_does_not_trigger_alert(self):
        context = {"x": 1.0}
        options = [{"name": "A", "x": 100.0}, {"name": "B", "x": 1.0}]
        self.engine.analyze_decision(context, options)
        alerts = self.engine.get_alerts(status="open")
        self.assertEqual(len(alerts), 0)


class TestOutcomeLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "decisions.db")
        self.engine = DecisionIntelligenceEngine(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_outcome_matching_recommendation(self):
        context = {"x": 1.0}
        options = [{"name": "A", "x": 5}, {"name": "B", "x": 1}]
        result = self.engine.analyze_decision(context, options)
        self.engine.log_outcome(result["decision_id"], "A", actual_outcome=1.0)
        metrics = self.engine.get_performance_metrics()
        self.assertEqual(metrics["total_outcomes_logged"], 1)
        self.assertEqual(metrics["matched_recommendation_count"], 1)
        self.assertEqual(metrics["recommendation_match_rate"], 1.0)

    def test_log_outcome_not_matching_recommendation(self):
        context = {"x": 1.0}
        options = [{"name": "A", "x": 5}, {"name": "B", "x": 1}]
        result = self.engine.analyze_decision(context, options)
        self.engine.log_outcome(result["decision_id"], "B", actual_outcome=0.5)
        metrics = self.engine.get_performance_metrics()
        self.assertEqual(metrics["matched_recommendation_count"], 0)
        self.assertEqual(metrics["recommendation_match_rate"], 0.0)

    def test_performance_metrics_empty_state(self):
        metrics = self.engine.get_performance_metrics()
        self.assertEqual(metrics["total_outcomes_logged"], 0)
        self.assertEqual(metrics["recommendation_match_rate"], 0.0)


class TestAlerts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "decisions.db")
        self.engine = DecisionIntelligenceEngine(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_trigger_and_resolve_alert(self):
        alert_id = self.engine.trigger_alert("manual", "medium", "Test alert")
        open_alerts = self.engine.get_alerts(status="open")
        self.assertEqual(len(open_alerts), 1)

        resolved = self.engine.resolve_alert(alert_id)
        self.assertTrue(resolved)

        open_after = self.engine.get_alerts(status="open")
        self.assertEqual(len(open_after), 0)

    def test_resolve_nonexistent_alert_returns_false(self):
        resolved = self.engine.resolve_alert(9999)
        self.assertFalse(resolved)


class TestDashboardSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "decisions.db")
        self.engine = DecisionIntelligenceEngine(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dashboard_summary_counts_are_accurate(self):
        context = {"x": 1.0}
        options = [{"name": "A", "x": 5}, {"name": "B", "x": 1}]
        result = self.engine.analyze_decision(context, options)
        self.engine.log_outcome(result["decision_id"], "A", actual_outcome=1.0)
        self.engine.trigger_alert("manual", "low", "extra alert")

        summary = self.engine.get_dashboard_summary()
        self.assertEqual(summary["total_decisions"], 1)
        self.assertEqual(summary["total_outcomes_logged"], 1)
        self.assertEqual(summary["active_alerts"], 1)

    def test_dashboard_summary_empty_state(self):
        summary = self.engine.get_dashboard_summary()
        self.assertEqual(summary["total_decisions"], 0)
        self.assertEqual(summary["average_confidence"], 0.0)
        self.assertEqual(summary["total_outcomes_logged"], 0)
        self.assertEqual(summary["active_alerts"], 0)


if __name__ == "__main__":
    unittest.main()
