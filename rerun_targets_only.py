from __future__ import annotations

from pathlib import Path

from target3d.audit import run_audit
from target3d.common import load_config, setup_logging
from target3d.reporting import run_reporting
from target3d.targets import run_targets


def main() -> None:
    setup_logging()
    cfg = load_config("config.yaml")
    audit = run_audit(cfg)
    model_dir = Path("outputs/01_modeling")
    required = ("mu.npy", "u.npy", "robust_scores.parquet", "threshold_summary.csv")
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing completed modeling artifacts: {missing}")
    modeling = {
        "mu": model_dir / "mu.npy",
        "u": model_dir / "u.npy",
        "robust_scores": model_dir / "robust_scores.parquet",
        "thresholds": model_dir / "threshold_summary.csv",
    }
    run_targets(cfg, audit, modeling)
    report = run_reporting(cfg)
    print(f"Complete: {report['analysis_bundle']}")


if __name__ == "__main__":
    main()
