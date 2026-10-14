import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_test_spec, iter_snapshots

logger = logging.getLogger(__name__)

STAGE_LABELS = ("Normal", "Early_Degradation", "Severe_Fault")
NON_FEATURE_COLUMNS = ("timestamp", "run_progress", "stage")

_EPS = 1e-12


def compute_time_features(frame: pd.DataFrame) -> Dict[str, float]:
    """Collapse one 20,480-sample snapshot into seven descriptors per channel.

    The raw record is far too long to feed a classifier directly, and the
    waveform itself is phase-random from one snapshot to the next. Summary
    statistics discard the phase and keep what actually changes as a bearing
    wears: how much energy is present and how impulsive it is.
    """
    x = frame.to_numpy(dtype=np.float64)
    centered_mean = x.mean(axis=0)
    centered = x - centered_mean

    # Mean: the accelerometer should sit near zero. A drifting mean points at
    # sensor bias or a static preload shift rather than at bearing damage, so
    # it is kept as a sanity channel the model can learn to ignore.
    mean = centered_mean

    # Std (population): AC energy around the DC offset. Rises smoothly as
    # spalling grows and every ball pass injects more broadband vibration.
    std = centered.std(axis=0)

    # RMS: square root of mean power, i.e. the overall energy the housing is
    # absorbing. It reacts late -- a single small defect barely moves it --
    # but it climbs monotonically once damage is widespread, which makes it
    # the most reliable indicator of the severe stage.
    rms = np.sqrt(np.mean(x ** 2, axis=0))

    peak = np.abs(x).max(axis=0)

    # Peak-to-peak: full swing of the signal. Dominated by the single worst
    # impact in the window, so it jumps earlier than RMS when a defect first
    # starts striking the rolling elements, but it is also the noisiest.
    peak_to_peak = x.max(axis=0) - x.min(axis=0)

    safe_std = np.where(std > _EPS, std, np.nan)

    # Skewness: asymmetry of the amplitude distribution. A healthy bearing is
    # close to symmetric noise; a localized defect hit in one direction biases
    # the acceleration pulses and pushes skewness away from zero.
    skewness = np.mean(centered ** 3, axis=0) / safe_std ** 3

    # Kurtosis (Pearson, Gaussian = 3): fourth-moment measure of tail weight.
    # Early in a fault the signal is mostly noise with a few sharp impact
    # transients, so energy barely changes but the tails fatten and kurtosis
    # spikes. Once damage spreads, impacts blur into continuous roughness and
    # kurtosis falls back toward 3 while RMS keeps rising; that crossover is
    # exactly why both are needed.
    kurtosis = np.mean(centered ** 4, axis=0) / safe_std ** 4

    # Crest factor: peak / RMS. Normalizing the worst spike by the baseline
    # energy makes it independent of load and sensor gain, so it isolates
    # impulsiveness. It is high when isolated impacts ride on a quiet signal
    # and decays as RMS catches up with the peaks.
    crest_factor = peak / np.where(rms > _EPS, rms, np.nan)

    stats = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "peak_to_peak": peak_to_peak,
    }

    features: Dict[str, float] = {}
    for stat_name, values in stats.items():
        for channel, value in zip(frame.columns, values):
            # A dead or clipped channel has zero variance and would produce
            # NaN moments; zero is a neutral fill that trees can isolate.
            features[f"{channel}_{stat_name}"] = float(np.nan_to_num(value, nan=0.0))
    return features


def assign_stages(
    timestamps: pd.Series,
    normal_until: float = 0.60,
    early_until: float = 0.85,
) -> pd.DataFrame:
    """Map acquisition time linearly onto run progress and a health stage.

    IMS runs end at failure, so elapsed fraction of the run is a usable proxy
    for remaining useful life. Cut points are fractions of total run length:
    [0, normal_until) Normal, [normal_until, early_until) Early_Degradation,
    the remainder Severe_Fault.
    """
    if not 0.0 < normal_until < early_until < 1.0:
        raise ValueError("Stage thresholds must satisfy 0 < normal_until < early_until < 1")

    elapsed = (timestamps - timestamps.min()).dt.total_seconds()
    span = elapsed.max()
    if span <= 0:
        raise ValueError("Cannot assign stages: all snapshots share one timestamp")

    progress = elapsed / span
    stage = np.select(
        [progress < normal_until, progress < early_until],
        [STAGE_LABELS[0], STAGE_LABELS[1]],
        default=STAGE_LABELS[2],
    )
    return pd.DataFrame({"run_progress": progress.to_numpy(), "stage": stage})


def build_feature_matrix(
    data_dir: Path,
    test_set: int,
    max_files: Optional[int] = None,
    normal_until: float = 0.60,
    early_until: float = 0.85,
) -> pd.DataFrame:
    get_test_spec(test_set)

    rows: List[Dict[str, float]] = []
    timestamps: List[pd.Timestamp] = []

    # One snapshot in memory at a time; only the compact feature dict is kept.
    for timestamp, frame in iter_snapshots(data_dir, test_set, max_files=max_files):
        rows.append(compute_time_features(frame))
        timestamps.append(pd.Timestamp(timestamp))
        if len(rows) % 250 == 0:
            logger.info("Processed %d snapshots (latest %s)", len(rows), timestamp)

    if not rows:
        raise RuntimeError(f"No valid snapshots could be processed from {data_dir}")

    matrix = pd.DataFrame(rows)
    matrix.insert(0, "timestamp", pd.Series(timestamps))
    labels = assign_stages(matrix["timestamp"], normal_until, early_until)
    matrix = pd.concat([matrix, labels], axis=1)

    logger.info("Feature matrix built: %d rows x %d columns", *matrix.shape)
    return matrix


def save_feature_matrix(matrix: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pyarrow (pip install pyarrow)") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not write feature matrix to {output_path}: {exc}") from exc

    size_kb = output_path.stat().st_size / 1024
    logger.info("Saved %s (%.1f KB)", output_path, size_kb)
    return output_path


def run_pipeline(
    data_dir: Path,
    test_set: int,
    output_path: Path,
    max_files: Optional[int] = None,
    normal_until: float = 0.60,
    early_until: float = 0.85,
) -> Path:
    matrix = build_feature_matrix(data_dir, test_set, max_files, normal_until, early_until)
    counts = matrix["stage"].value_counts().reindex(STAGE_LABELS, fill_value=0)
    logger.info("Stage distribution: %s", counts.to_dict())
    return save_feature_matrix(matrix, output_path)