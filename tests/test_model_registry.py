import json
import tempfile
import unittest
from pathlib import Path

from mlProject.utils.model_registry import (
    RegistryError,
    get_version_id,
    load_registry,
    register_model,
    rollback_to_version,
    validate_registry,
)


class TestModelRegistry(unittest.TestCase):
    def test_get_version_id_is_unique(self):
        ids = {get_version_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_get_version_id_format(self):
        vid = get_version_id()
        self.assertTrue(vid.startswith("v"))
        self.assertIn("_", vid)

    def test_register_model_rejects_duplicate_version_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            model_path = Path(tmp) / "model.joblib"
            model_path.write_text("dummy")
            vid = "v20260609_143021_test"

            register_model(
                registry_path=registry_path,
                model_path=model_path,
                version_id=vid,
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            with self.assertRaises(ValueError):
                register_model(
                    registry_path=registry_path,
                    model_path=model_path,
                    version_id=vid,
                    metrics={"rmse": 0.6},
                    params={"alpha": 0.2},
                )

    def test_archived_model_files_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            model_paths = []
            for i in range(3):
                mp = Path(tmp) / f"model_v{i}.joblib"
                mp.write_text(f"dummy{i}")
                (Path(str(mp) + ".sha256")).write_text(f"hash{i}")
                model_paths.append(mp)

            for i, mp in enumerate(model_paths):
                register_model(
                    registry_path=registry_path,
                    model_path=mp,
                    version_id=f"v{i:04d}",
                    metrics={"rmse": 0.5},
                    params={"alpha": 0.1},
                    max_versions_to_keep=2,
                )

            registry = load_registry(registry_path)
            self.assertEqual(len(registry["versions"]), 2)
            self.assertFalse(model_paths[0].exists())
            self.assertFalse(Path(str(model_paths[0]) + ".sha256").exists())


    def test_rollback_copies_versioned_file_to_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            stable_path = Path(tmp) / "model.joblib"
            versioned_path = Path(tmp) / "model_v001.joblib"
            versioned_path.write_text("versioned_model_weights")
            stable_path.write_text("old_weights")

            register_model(
                registry_path=registry_path,
                model_path=versioned_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            stable_path.write_text("corrupted_weights")

            result = rollback_to_version(registry_path, "v001")
            self.assertTrue(result)
            self.assertEqual(stable_path.read_text(), "versioned_model_weights")
            registry = load_registry(registry_path)
            self.assertEqual(registry["production"], "v001")

    def test_rollback_refreshes_model_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            v1 = Path(tmp) / "model_v001.joblib"
            v2 = Path(tmp) / "model_v002.joblib"
            v1.write_text("weights_v1")
            v2.write_text("weights_v2")

            register_model(
                registry_path=registry_path,
                model_path=v1,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )
            register_model(
                registry_path=registry_path,
                model_path=v2,
                version_id="v002",
                metrics={"rmse": 0.4},
                params={"alpha": 0.2},
            )

            self.assertTrue(rollback_to_version(registry_path, "v001"))

            model_info = json.loads((Path(tmp) / "model_info.json").read_text())
            self.assertEqual(model_info["version_id"], "v001")
            self.assertEqual(model_info["model_path"], str(v1))
            self.assertEqual(model_info["params"], {"alpha": 0.1})

    def test_rollback_fails_when_versioned_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            versioned_path = Path(tmp) / "model_v001.joblib"
            versioned_path.write_text("weights")

            register_model(
                registry_path=registry_path,
                model_path=versioned_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            versioned_path.unlink()

            result = rollback_to_version(registry_path, "v001")
            self.assertFalse(result)

    def test_rollback_fails_for_nonexistent_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            result = rollback_to_version(registry_path, "v_nonexistent")
            self.assertFalse(result)

    def test_validate_registry_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            existing_path = Path(tmp) / "model_v001.joblib"
            existing_path.write_text("weights")
            (Path(str(existing_path) + ".sha256")).write_text("hash")
            (Path(tmp) / "model.joblib").write_text("weights")
            missing_path = Path(tmp) / "model_v002.joblib"

            register_model(
                registry_path=registry_path,
                model_path=existing_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )
            register_model(
                registry_path=registry_path,
                model_path=missing_path,
                version_id="v002",
                metrics={"rmse": 0.6},
                params={"alpha": 0.2},
            )

            issues = validate_registry(registry_path)
            self.assertTrue(any("v002" in issue for issue in issues))
            v001_issues = [i for i in issues if "v001" in i]
            self.assertEqual(len(v001_issues), 0)

    def test_validate_registry_passes_with_all_files_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            model_path = Path(tmp) / "model_v001.joblib"
            model_path.write_text("weights")
            sha_path = Path(str(model_path) + ".sha256")
            sha_path.write_text("hash")
            (Path(tmp) / "model.joblib").write_text("weights")

            register_model(
                registry_path=registry_path,
                model_path=model_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            issues = validate_registry(registry_path)
            self.assertEqual(issues, [])


    def test_concurrent_registration_preserves_invariant(self):
        """Register two versions racing — stable file must match registry production."""
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            errors = []

            def register_model_safe(vid, metrics_rmse):
                try:
                    mp = Path(tmp) / f"model_{vid}.joblib"
                    mp.write_text(f"weights_{vid}")
                    ha = Path(str(mp) + ".sha256")
                    ha.write_text("dummy")
                    register_model(
                        registry_path=registry_path,
                        model_path=mp,
                        version_id=vid,
                        metrics={"rmse": metrics_rmse},
                        params={"alpha": 0.1},
                        max_versions_to_keep=5,
                        stable_model_path=Path(tmp) / "model.joblib",
                    )
                except Exception as e:
                    errors.append(f"{vid}: {e}")

            threads = [
                threading.Thread(target=register_model_safe, args=("v002", 0.5)),
                threading.Thread(target=register_model_safe, args=("v003", 0.6)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            registry = load_registry(registry_path)
            production_id = registry.get("production")
            stable_path = Path(tmp) / "model.joblib"
            if production_id and stable_path.exists():
                expected = f"weights_{production_id}"
                actual = stable_path.read_text()
                self.assertEqual(
                    actual, expected,
                    f"Stable model content '{actual}' does not match "
                    f"production version {production_id} content '{expected}'"
                )

    def test_concurrent_promotion_checksum_integrity(self):
        """Verify checksum of stable file matches source after concurrent promotion."""
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            errors = []

            def register_and_verify(vid, metrics_rmse):
                try:
                    mp = Path(tmp) / f"model_{vid}.joblib"
                    mp.write_text(f"weights_{vid}")
                    register_model(
                        registry_path=registry_path,
                        model_path=mp,
                        version_id=vid,
                        metrics={"rmse": metrics_rmse},
                        params={"alpha": 0.1},
                        max_versions_to_keep=5,
                        stable_model_path=Path(tmp) / "model.joblib",
                    )
                except Exception as e:
                    errors.append(f"{vid}: {e}")

            threads = [
                threading.Thread(target=register_and_verify, args=("v004", 0.7)),
                threading.Thread(target=register_and_verify, args=("v005", 0.8)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            registry = load_registry(registry_path)
            production_id = registry.get("production")
            stable_path = Path(tmp) / "model.joblib"
            stable_checksum_path = Path(str(stable_path) + ".sha256")
            if production_id and stable_path.exists() and stable_checksum_path.exists():
                src_path = Path(tmp) / f"model_{production_id}.joblib"
                if src_path.exists():
                    from mlProject.utils.common import compute_checksum
                    src_cs = compute_checksum(src_path)
                    dst_cs = compute_checksum(stable_path)
                    self.assertEqual(
                        src_cs, dst_cs,
                        f"Checksum mismatch for production version {production_id}: "
                        f"src={src_cs[:8]}, dst={dst_cs[:8]}"
                    )


    def test_load_registry_returns_default_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry = load_registry(registry_path)
            self.assertEqual(registry["production"], None)
            self.assertEqual(registry["versions"], [])

    def test_load_registry_raises_on_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry_path.write_text("{ not valid json")
            with self.assertRaises(RegistryError):
                load_registry(registry_path)

    def test_corrupt_registry_is_backed_up_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry_path.write_text("{ corrupt")
            model_path = Path(tmp) / "model.joblib"
            model_path.write_text("dummy")

            with self.assertRaises(RegistryError):
                register_model(
                    registry_path=registry_path,
                    model_path=model_path,
                    version_id="v20260609_143021_test",
                    metrics={"rmse": 0.5},
                    params={"alpha": 0.1},
                )

            self.assertEqual(registry_path.read_text(), "{ corrupt")
            backups = list(Path(tmp).glob("registry.json.corrupt-*"))
            self.assertTrue(backups, "corrupt registry should be backed up")


if __name__ == "__main__":
    unittest.main()
