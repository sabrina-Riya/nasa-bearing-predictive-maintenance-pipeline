

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SAMPLING_RATE_HZ = 20_000
SAMPLES_PER_SNAPSHOT = 20_480

_TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"
_TIMESTAMP_PATTERN = re.compile(r"^\d{4}(\.\d{2}){5}$")


class SnapshotReadError(Exception):
    """Raised when a snapshot file is unreadable or structurally invalid."""


@dataclass(frozen=True)
class BearingTestSpec:
    """Static description of one IMS test-to-failure experiment."""

    test_set: int
    channels: Tuple[str, ...]
    failed_bearing: int
    failure_mode: str


# Test 1 instrumented each bearing with two accelerometers (x/y axes);
# tests 2 and 3 used a single accelerometer per bearing. Failure labels
# below are taken from the dataset readme and are what we derive the
# health-stage targets from downstream.
TEST_SPECS = {
    1: BearingTestSpec(
        test_set=1,
        channels=tuple(f"b{b}_{axis}" for b in range(1, 5) for axis in ("x", "y")),
        failed_bearing=3,
        failure_mode="inner_race",
    ),
    2: BearingTestSpec(
        test_set=2,
        channels=tuple(f"b{b}" for b in range(1, 5)),
        failed_bearing=1,
        failure_mode="outer_race",
    ),
    3: BearingTestSpec(
        test_set=3,
        channels=tuple(f"b{b}" for b in range(1, 5)),
        failed_bearing=3,
        failure_mode="outer_race",
    ),
}


def get_test_spec(test_set: int) -> BearingTestSpec:
    try:
        return TEST_SPECS[test_set]
    except KeyError:
        raise ValueError(
            f"Unknown IMS test set {test_set!r}; expected one of {sorted(TEST_SPECS)}"
        ) from None


def parse_snapshot_timestamp(path: Path) -> datetime:
    """Convert a snapshot file name such as 2003.10.22.12.06.24 to a datetime."""
    path = Path(path)
    if not _TIMESTAMP_PATTERN.match(path.name):
        raise ValueError(f"'{path.name}' is not a valid IMS snapshot file name")
    return datetime.strptime(path.name, _TIMESTAMP_FORMAT)


def list_snapshot_files(data_dir: Path) -> List[Path]:
    """Return snapshot files under data_dir in chronological order."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    snapshots = []
    for entry in data_dir.iterdir():
        if not entry.is_file():
            continue
        try:
            snapshots.append((parse_snapshot_timestamp(entry), entry))
        except ValueError:
            logger.debug("Ignoring non-snapshot file %s", entry.name)

    if not snapshots:
        raise FileNotFoundError(f"No IMS snapshot files found in {data_dir}")

    snapshots.sort(key=lambda item: item[0])
    return [path for _, path in snapshots]


def load_snapshot(path: Path, spec: BearingTestSpec) -> pd.DataFrame:
    """Read one snapshot into a (20480, n_channels) frame with named columns."""
    path = Path(path)
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, dtype="float64")
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SnapshotReadError(f"Could not read {path.name}: {exc}") from exc

    if frame.shape[1] != len(spec.channels):
        raise SnapshotReadError(
            f"{path.name}: expected {len(spec.channels)} channels, found {frame.shape[1]}"
        )
    # A short record would skew spectral resolution (bin width = fs / N),
    # so it is rejected instead of being silently fed into the FFT stage.
    if len(frame) != SAMPLES_PER_SNAPSHOT:
        raise SnapshotReadError(
            f"{path.name}: expected {SAMPLES_PER_SNAPSHOT} samples, found {len(frame)}"
        )
    if frame.isna().any().any():
        raise SnapshotReadError(f"{path.name}: contains missing values")

    frame.columns = list(spec.channels)
    return frame


def iter_snapshots(
    data_dir: Path,
    test_set: int,
    max_files: Optional[int] = None,
) -> Iterator[Tuple[datetime, pd.DataFrame]]:
    """Yield (timestamp, signal frame) one file at a time, oldest first.

    Only a single snapshot is held in memory at any point, which mirrors how
    a live acquisition system would hand data to the feature stage. Corrupt
    files are logged and skipped so one bad record cannot abort a run that
    spans thousands of files.
    """
    spec = get_test_spec(test_set)
    files = list_snapshot_files(data_dir)
    if max_files is not None:
        files = files[:max_files]

    skipped = 0
    for path in files:
        try:
            frame = load_snapshot(path, spec)
        except SnapshotReadError as exc:
            skipped += 1
            logger.warning("Skipping snapshot: %s", exc)
            continue
        yield parse_snapshot_timestamp(path), frame

    if skipped:
        logger.warning("%d of %d snapshots were skipped", skipped, len(files))