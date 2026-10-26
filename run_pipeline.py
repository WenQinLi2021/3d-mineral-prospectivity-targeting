from __future__ import annotations

import argparse
import sys
from pathlib import Path

from target3d.audit import run_audit
from target3d.common import LOG, load_config, setup_logging
from target3d.modeling import run_modeling
from target3d.reporting import run_reporting
from target3d.targets import run_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3D target pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--through",
        choices=["audit", "modeling", "targets", "report"],
        default="report",
        help="Stop after this stage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    audit = run_audit(cfg)
    if args.through == "audit":
        return 0
    modeling = run_modeling(cfg, audit)
    if args.through == "modeling":
        return 0
    run_targets(cfg, audit, modeling)
    if args.through == "targets":
        return 0
    report = run_reporting(cfg)
    LOG.info("Pipeline complete. Download %s", report["analysis_bundle"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
