import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import run_pipeline
from src.train import load_feature_matrix, train_and_evaluate
from src.utils import TEST_SPECS

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_FILE = PROJECT_ROOT / "models" / "rf_bearing_health.joblib"
REPORT_FILE = PROJECT_ROOT / "reports" / "classification_report.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA IMS bearing health-stage pipeline")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "2nd_test")
    parser.add_argument("--test-set", type=int, choices=sorted(TEST_SPECS), default=2)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--normal-until", type=float, default=0.60)
    parser.add_argument("--early-until", type=float, default=0.85)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="reuse an existing features.parquet instead of re-reading raw files")
    return parser.parse_args()


def ensure_directories(data_dir: Path, need_raw: bool) -> None:
    if need_raw and not data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")
    for target in (FEATURES_FILE.parent, MODEL_FILE.parent, REPORT_FILE.parent):
        target.mkdir(parents=True, exist_ok=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    log = logging.getLogger("main")
    args = parse_args()

    try:
        ensure_directories(args.data_dir, need_raw=not args.skip_extraction)

        if args.skip_extraction:
            log.info("Reusing %s", FEATURES_FILE)
        else:
            run_pipeline(
                data_dir=args.data_dir,
                test_set=args.test_set,
                output_path=FEATURES_FILE,
                max_files=args.max_files,
                normal_until=args.normal_until,
                early_until=args.early_until,
            )

        matrix = load_feature_matrix(FEATURES_FILE)
        train_and_evaluate(
            matrix,
            model_path=MODEL_FILE,
            report_path=REPORT_FILE,
            train_fraction=args.train_fraction,
            gap=args.gap,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())