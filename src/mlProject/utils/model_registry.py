import json
import os
import hashlib
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import portalocker

from mlProject import logger


# Registry version stamp for optimistic concurrency
_REGISTRY_VERSION_KEY = "_version_stamp"


class RegistryError(Exception):
    """Raised when an existing registry file is present but cannot be read."""


def _next_version_stamp() -> str:
    return datetime.now(timezone.utc).isoformat() + ":" + uuid.uuid4().hex[:8]


def _lock_registry(registry_path: Path):
    """Acquire an exclusive lock on the registry file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")

    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        portalocker.lock(lock_file, portalocker.LOCK_EX)
        return lock_file
    except Exception:
        if lock_file is not None:
            lock_file.close()
        raise


def _unlock_registry(lock_file):
    """Release the registry lock."""
    portalocker.unlock(lock_file)
    lock_file.close()


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def load_registry(registry_path: Path) -> dict:
    """Load model registry from JSON file.

    A missing file returns an empty registry. An existing file that cannot be
    parsed is treated as corruption and raises RegistryError, so that callers
    that load-mutate-save do not overwrite a recoverable file with an empty one.
    The unreadable file is side-copied for recovery before raising.
    """
    if not registry_path.exists():
        return {"production": None, "staging": None, "versions": []}
    try:
        with open(registry_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        backup_path = registry_path.with_suffix(
            registry_path.suffix + f".corrupt-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            shutil.copy2(str(registry_path), str(backup_path))
            logger.error(f"Failed to load registry: {e}. Backed up to {backup_path}")
        except OSError:
            logger.error(f"Failed to load registry: {e}. Backup also failed.")
        raise RegistryError(f"Registry at {registry_path} is unreadable: {e}") from e


def save_registry(registry_path: Path, registry: dict):
    """Atomically save model registry to JSON file under a file lock."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp with a version for optimistic concurrency detection
    registry[_REGISTRY_VERSION_KEY] = _next_version_stamp()
    # Atomic write: write to temp file, then replace atomically
    fd, tmp_path = tempfile.mkstemp(
        dir=registry_path.parent,
        suffix=".tmp",
        prefix=registry_path.stem + "_",
    )
    try:
        with os.fdopen(fd, "w") as tmp:
            json.dump(registry, tmp, indent=2)
        os.replace(tmp_path, str(registry_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    logger.info(f"Model registry saved to {registry_path}")


def get_version_id() -> str:
    """Generate a globally unique version ID using timestamp and UUID."""
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    suffix = uuid.uuid4().hex[:8]
    return f"v{ts}_{suffix}"


def register_model(
    registry_path: Path,
    model_path: Path,
    version_id: str,
    metrics: dict,
    params: dict,
    data_hash: Optional[str] = None,
    max_versions_to_keep: int = 10,
    quality_gate_max_rmse_degradation_pct: float = 5.0,
    stable_model_path: Optional[Path] = None,
) -> dict:
    """Register a model version and enforce quality gates."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot register model {version_id}: model file not found at {model_path}"
        )
    lock = _lock_registry(registry_path)
    try:
        registry = load_registry(registry_path)

        for v in registry.get("versions", []):
            if v.get("id") == version_id:
                raise ValueError(f"Version ID {version_id} already exists in registry")

        if not metrics:
            logger.warning(
                f"Model {version_id} registered with empty metrics — "
                "skipping quality gate, setting status to 'pending'"
            )
            entry = {
                "id": version_id,
                "path": str(model_path),
                "metrics": metrics,
                "params": params,
                "date": datetime.now(timezone.utc).isoformat(),
                "data_hash": data_hash or "",
                "status": "pending",
            }
            registry["versions"].insert(0, entry)
            if len(registry["versions"]) > max_versions_to_keep:
                archived = registry["versions"][max_versions_to_keep:]
                registry["versions"] = registry["versions"][:max_versions_to_keep]
                for v in archived:
                    archived_path = Path(v["path"])
                    if archived_path.exists():
                        archived_path.unlink()
                        logger.info(f"Deleted archived model file: {archived_path}")
                    sha_path = Path(str(archived_path) + ".sha256")
                    if sha_path.exists():
                        sha_path.unlink()
                        logger.info(f"Deleted archived checksum: {sha_path}")
            save_registry(registry_path, registry)
            return entry

        if "rmse" not in metrics:
            raise ValueError(
                f"Cannot register model {version_id}: metrics dict must contain 'rmse' key. "
                f"Got keys: {list(metrics.keys())}"
            )

        current_production = registry.get("production")
        previous_metrics = None
        if current_production:
            for v in registry.get("versions", []):
                if v.get("id") == current_production:
                    previous_metrics = v.get("metrics", {})
                    break

        status = "staging"
        if previous_metrics and "rmse" in previous_metrics:
            prev_rmse = previous_metrics["rmse"]
            new_rmse = metrics["rmse"]
            if prev_rmse > 0:
                degradation_pct = ((new_rmse - prev_rmse) / prev_rmse) * 100
                if degradation_pct > quality_gate_max_rmse_degradation_pct:
                    status = "rejected"
                    logger.warning(
                        f"Model {version_id} REJECTED: RMSE degradation {degradation_pct:.2f}% "
                        f"exceeds threshold {quality_gate_max_rmse_degradation_pct}%"
                    )
                else:
                    status = "production"
                    registry["production"] = version_id
                    logger.info(
                        f"Model {version_id} PROMOTED to production: "
                        f"RMSE degradation {degradation_pct:.2f}% within threshold"
                    )
            else:
                status = "production"
                registry["production"] = version_id
                logger.info(f"Model {version_id} registered as production (previous RMSE was 0)")
        else:
            status = "production"
            registry["production"] = version_id
            logger.info(f"First model {version_id} registered as production")

        # Two-phase promotion: copy model to stable path under lock before marking as production
        if status == "production" and stable_model_path is not None:
            stable_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(model_path), str(stable_model_path))
            from mlProject.utils.common import compute_checksum, save_checksum
            src_checksum = compute_checksum(model_path)
            dst_checksum = compute_checksum(stable_model_path)
            if src_checksum != dst_checksum:
                status = "rejected"
                if registry.get("production") == version_id:
                    del registry["production"]
                logger.error(
                    f"Model {version_id} REJECTED: checksum mismatch after copy "
                    f"to stable path {stable_model_path} (src={src_checksum[:8]}, dst={dst_checksum[:8]})"
                )
            else:
                stable_checksum_path = Path(str(stable_model_path) + ".sha256")
                save_checksum(stable_model_path, stable_checksum_path)
                logger.info(
                    f"Model {version_id} copied to stable path {stable_model_path} "
                    f"and checksum verified ({src_checksum[:8]})"
                )

        entry = {
            "id": version_id,
            "path": str(model_path),
            "metrics": metrics,
            "params": params,
            "date": datetime.now(timezone.utc).isoformat(),
            "data_hash": data_hash or "",
            "status": status,
        }

        registry["versions"].insert(0, entry)

        if len(registry["versions"]) > max_versions_to_keep:
            archived = registry["versions"][max_versions_to_keep:]
            registry["versions"] = registry["versions"][:max_versions_to_keep]
            protected = {registry.get("production"), registry.get("staging")} - {None}
            for v in archived:
                if v.get("id") in protected:
                    logger.info(f"Skipping deletion of protected model {v.get('id')} (active production/staging alias)")
                    continue
                archived_path = Path(v["path"])
                if archived_path.exists():
                    archived_path.unlink()
                    logger.info(f"Deleted archived model file: {archived_path}")
                sha_path = Path(str(archived_path) + ".sha256")
                if sha_path.exists():
                    sha_path.unlink()
                    logger.info(f"Deleted archived checksum: {sha_path}")

        save_registry(registry_path, registry)
        return entry
    finally:
        _unlock_registry(lock)


def update_registration(
    registry_path: Path,
    version_id: str,
    metrics: dict = None,
    status: str = None,
    model_path: Path = None,
    params: dict = None,
    data_hash: str = None,
    quality_gate_max_rmse_degradation_pct: float = None,
    stable_model_path: Path = None,
) -> bool:
    """Update an existing registry entry's metrics, status, model path, params, and/or data hash."""
    lock = _lock_registry(registry_path)
    try:
        registry = load_registry(registry_path)
        for v in registry.get("versions", []):
            if v.get("id") == version_id:
                was_production = v.get("status") == "production" or registry.get("production") == version_id
                if metrics is not None:
                    v["metrics"] = metrics
                if status is not None:
                    v["status"] = status
                    if status == "production":
                        registry["production"] = version_id
                        # demote previous production if any
                        for other_v in registry.get("versions", []):
                            if other_v.get("id") != version_id and other_v.get("status") == "production":
                                other_v["status"] = "staging"
                    elif status == "staging":
                        if registry.get("production") == version_id:
                            registry["production"] = None
                        registry["staging"] = version_id
                    elif status == "archived" or status == "rejected":
                        if registry.get("production") == version_id:
                            registry["production"] = None
                        if registry.get("staging") == version_id:
                            registry["staging"] = None
                if model_path is not None:
                    v["path"] = str(model_path)
                if params is not None:
                    v["params"] = params
                if data_hash is not None:
                    v["data_hash"] = data_hash
                if quality_gate_max_rmse_degradation_pct is not None:
                    v["quality_gate_max_rmse_degradation_pct"] = quality_gate_max_rmse_degradation_pct
                v["updated_at"] = datetime.now(timezone.utc).isoformat()

                # If this version is production, ensure stable model file is in sync
                is_production = v.get("status") == "production" or registry.get("production") == version_id
                if is_production and stable_model_path is not None:
                    source_path = Path(model_path) if model_path is not None else Path(v["path"])
                    if source_path.exists():
                        stable_model_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(source_path), str(stable_model_path))
                        from mlProject.utils.common import compute_checksum, save_checksum
                        src_checksum = compute_checksum(source_path)
                        dst_checksum = compute_checksum(stable_model_path)
                        if src_checksum != dst_checksum:
                            logger.error(
                                f"Model {version_id} stable copy checksum mismatch during update: "
                                f"src={src_checksum[:8]}, dst={dst_checksum[:8]}"
                            )
                        else:
                            stable_checksum_path = Path(str(stable_model_path) + ".sha256")
                            save_checksum(stable_model_path, stable_checksum_path)
                            logger.info(
                                f"Model {version_id} stable copy synced during update "
                                f"({src_checksum[:8]})"
                            )
                    else:
                        logger.warning(
                            f"Source model file {source_path} not found for production "
                            f"version {version_id} — stable copy skipped"
                        )

                save_registry(registry_path, registry)
                logger.info(
                    f"Updated registration for version {version_id}: "
                    f"metrics={metrics is not None}, status={status}"
                )
                return True
        return False
    finally:
        _unlock_registry(lock)


def get_production_model_path(registry_path: Path) -> Optional[Path]:
    """Get the production model path from the registry."""
    registry = load_registry(registry_path)
    production_id = registry.get("production")
    if production_id:
        for v in registry.get("versions", []):
            if v.get("id") == production_id:
                return Path(v["path"])
    return None


def get_staging_model_path(registry_path: Path) -> Optional[Path]:
    """Get the staging model path from the registry."""
    registry = load_registry(registry_path)
    staging_id = registry.get("staging")
    if staging_id:
        for v in registry.get("versions", []):
            if v.get("id") == staging_id:
                return Path(v["path"])
    return None


def rollback_to_version(registry_path: Path, version_id: str) -> bool:
    """Rollback production alias to a specific version and restore the model file."""
    from mlProject.utils.common import save_checksum
    lock = _lock_registry(registry_path)
    try:
        registry = load_registry(registry_path)
        for v in registry.get("versions", []):
            if v.get("id") == version_id:
                versioned_path = Path(v["path"])
                if not versioned_path.exists():
                    logger.error(
                        f"Cannot rollback to version {version_id}: "
                        f"model file not found at {versioned_path}"
                    )
                    return False
                stable_path = versioned_path.parent / "model.joblib"
                shutil.copy2(str(versioned_path), str(stable_path))
                checksum_path = Path(str(stable_path) + ".sha256")
                save_checksum(stable_path, checksum_path)
                model_info_path = stable_path.parent / "model_info.json"
                model_info = {
                    "version_id": version_id,
                    "model_path": str(versioned_path),
                    "params": v.get("params", {}),
                    "data_hash": v.get("data_hash", ""),
                }
                with open(model_info_path, "w") as f:
                    json.dump(model_info, f, indent=2)
                registry["production"] = version_id
                save_registry(registry_path, registry)
                logger.info(
                    f"Rolled back production to version {version_id}, "
                    f"restored model from {versioned_path} to {stable_path}"
                )
                return True
        logger.error(f"Version {version_id} not found in registry")
        return False
    finally:
        _unlock_registry(lock)


def validate_registry(registry_path: Path) -> List[str]:
    """Check whether all registered versions have corresponding files on disk."""
    registry = load_registry(registry_path)
    issues = []
    for v in registry.get("versions", []):
        version_path = Path(v["path"])
        if not version_path.exists():
            issues.append(f"Missing model file for version {v['id']}: {v['path']}")
        sha_path = Path(str(version_path) + ".sha256")
        if not sha_path.exists():
            issues.append(f"Missing checksum for version {v['id']}: {sha_path}")
        if v.get("status") == "production":
            production_path = version_path.parent / "model.joblib"
            if not production_path.exists():
                issues.append(
                    f"Production model file missing at {production_path} "
                    f"for version {v['id']}"
                )
    if issues:
        for issue in issues:
            logger.warning(f"Registry validation issue: {issue}")
    else:
        logger.info("Registry validation passed — all version files present")
    return issues