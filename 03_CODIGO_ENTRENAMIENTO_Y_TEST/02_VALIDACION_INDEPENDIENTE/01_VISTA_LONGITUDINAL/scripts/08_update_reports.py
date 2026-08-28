"""Regenera resumentes por experimento y reportes acumulados."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import EXPERIMENTS_ROOT
from src.reports import update_global_reports, write_experiment_summary


if __name__ == "__main__":
    for experiment_dir in sorted(path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()):
        config_path = experiment_dir / "config.json"
        validation_path = experiment_dir / "validation_metrics.csv"
        if not (config_path.exists() and validation_path.exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validation = pd.read_csv(validation_path).iloc[0].to_dict()
        test_path = experiment_dir / "test_metrics.csv"
        benchmark_path = experiment_dir / "benchmark_single_model.csv"
        test = pd.read_csv(test_path).iloc[0].to_dict() if test_path.exists() else None
        benchmark = pd.read_csv(benchmark_path).iloc[0].to_dict() if benchmark_path.exists() else None
        write_experiment_summary(experiment_dir, config, validation, test, benchmark)
        print(f"Resumen actualizado: {experiment_dir.name}")
    update_global_reports()
    print("Reportes acumulados actualizados.")
