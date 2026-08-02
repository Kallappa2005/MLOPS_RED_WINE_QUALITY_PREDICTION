import tempfile
import unittest
from pathlib import Path

from mlProject.components.ethical_monitoring import EthicalAIMonitor


class TestBiasDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "ethics.db")
        self.monitor = EthicalAIMonitor(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detect_bias_flags_disparity_above_threshold(self):
        records = (
            [{"group": "A", "prediction": 1} for _ in range(8)]
            + [{"group": "A", "prediction": 0} for _ in range(2)]
            + [{"group": "B", "prediction": 1} for _ in range(2)]
            + [{"group": "B", "prediction": 0} for _ in range(8)]
        )
        result = self.monitor.detect_bias(records, "group", threshold=0.1)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["bias_detected"])
        self.assertAlmostEqual(result["disparity"], 0.6, places=3)
        self.assertIn("recommendations", result)

    def test_detect_bias_no_flag_when_within_threshold(self):
        records = (
            [{"group": "A", "prediction": 1} for _ in range(5)]
            + [{"group": "A", "prediction": 0} for _ in range(5)]
            + [{"group": "B", "prediction": 1} for _ in range(5)]
            + [{"group": "B", "prediction": 0} for _ in range(5)]
        )
        result = self.monitor.detect_bias(records, "group", threshold=0.1)
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["bias_detected"])

    def test_detect_bias_requires_two_groups(self):
        records = [{"group": "A", "prediction": 1} for _ in range(5)]
        result = self.monitor.detect_bias(records, "group", threshold=0.1)
        self.assertEqual(result["status"], "insufficient_groups")

    def test_detect_bias_rejects_missing_fields(self):
        records = [{"group": "A"}]
        result = self.monitor.detect_bias(records, "group", threshold=0.1)
        self.assertEqual(result["status"], "error")

    def test_detect_bias_rejects_empty_records(self):
        result = self.monitor.detect_bias([], "group", threshold=0.1)
        self.assertEqual(result["status"], "error")

    def test_bias_check_is_persisted_to_history(self):
        records = (
            [{"group": "A", "prediction": 1} for _ in range(8)]
            + [{"group": "B", "prediction": 1} for _ in range(2)]
        )
        self.monitor.detect_bias(records, "group", threshold=0.1)
        history = self.monitor.get_bias_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["protected_attribute"], "group")

    def test_bias_detected_triggers_alert(self):
        records = (
            [{"group": "A", "prediction": 1} for _ in range(10)]
            + [{"group": "B", "prediction": 0} for _ in range(10)]
        )
        self.monitor.detect_bias(records, "group", threshold=0.1)
        alerts = self.monitor.get_alerts(status="open")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["alert_type"], "bias_detected")


class TestComplianceViolations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "ethics.db")
        self.monitor = EthicalAIMonitor(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_violation_appears_in_report(self):
        self.monitor.log_violation("data_bias", "medium", "Test violation")
        report = self.monitor.get_compliance_report()
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["violation_type"], "data_bias")
        self.assertEqual(report[0]["status"], "open")

    def test_high_severity_violation_triggers_alert(self):
        self.monitor.log_violation("fairness_breach", "critical", "Severe issue")
        alerts = self.monitor.get_alerts(status="open")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["alert_type"], "compliance_violation")

    def test_low_severity_violation_does_not_trigger_alert(self):
        self.monitor.log_violation("minor_issue", "low", "Not urgent")
        alerts = self.monitor.get_alerts(status="open")
        self.assertEqual(len(alerts), 0)


class TestAlerts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "ethics.db")
        self.monitor = EthicalAIMonitor(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_trigger_and_resolve_alert(self):
        alert_id = self.monitor.trigger_alert("manual", "medium", "Test alert")
        open_alerts = self.monitor.get_alerts(status="open")
        self.assertEqual(len(open_alerts), 1)

        resolved = self.monitor.resolve_alert(alert_id)
        self.assertTrue(resolved)

        open_alerts_after = self.monitor.get_alerts(status="open")
        self.assertEqual(len(open_alerts_after), 0)

        resolved_alerts = self.monitor.get_alerts(status="resolved")
        self.assertEqual(len(resolved_alerts), 1)

    def test_resolve_nonexistent_alert_returns_false(self):
        resolved = self.monitor.resolve_alert(9999)
        self.assertFalse(resolved)

    def test_get_alerts_all_status_returns_everything(self):
        self.monitor.trigger_alert("manual", "low", "One")
        alert_id = self.monitor.trigger_alert("manual", "low", "Two")
        self.monitor.resolve_alert(alert_id)
        all_alerts = self.monitor.get_alerts(status=None)
        self.assertEqual(len(all_alerts), 2)


class TestDashboardSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "ethics.db")
        self.monitor = EthicalAIMonitor(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dashboard_summary_counts_are_accurate(self):
        records = (
            [{"group": "A", "prediction": 1} for _ in range(10)]
            + [{"group": "B", "prediction": 0} for _ in range(10)]
        )
        self.monitor.detect_bias(records, "group", threshold=0.1)
        self.monitor.log_violation("issue", "critical", "desc")
        self.monitor.trigger_alert("manual", "low", "extra alert")

        summary = self.monitor.get_dashboard_summary()
        self.assertEqual(summary["total_bias_checks"], 1)
        self.assertEqual(summary["bias_flagged_checks"], 1)
        self.assertEqual(summary["open_violations"], 1)
        self.assertEqual(summary["active_alerts"], 3)

    def test_dashboard_summary_empty_state(self):
        summary = self.monitor.get_dashboard_summary()
        self.assertEqual(summary["total_bias_checks"], 0)
        self.assertEqual(summary["bias_flagged_checks"], 0)
        self.assertEqual(summary["open_violations"], 0)
        self.assertEqual(summary["active_alerts"], 0)


if __name__ == "__main__":
    unittest.main()
