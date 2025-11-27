"""File utility functions for artifact management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

ArtifactType = Literal["metrics", "checkpoints", "logs"]


class ArtifactPathResolver:
    """Resolves paths for training artifacts with fallback logic.

    The Tinker cookbook writes artifacts to a 'logs/' subdirectory,
    but simulation mode writes directly to the run directory.
    This class handles both cases transparently.
    """

    @staticmethod
    def get_artifact_path(
        run_log_path: str | Path,
        artifact_type: ArtifactType,
    ) -> Optional[Path]:
        """Get the path to an artifact file with fallback logic.

        Args:
            run_log_path: Path to the run's log file (e.g., artifacts/run_40/logs.txt)
            artifact_type: Type of artifact ('metrics', 'checkpoints', or 'logs')

        Returns:
            Path to the artifact file if it exists, None otherwise

        Examples:
            >>> # For real cookbook runs
            >>> get_artifact_path("artifacts/run_40/logs.txt", "metrics")
            Path("artifacts/run_40/logs/metrics.jsonl")

            >>> # For simulation runs (fallback)
            >>> get_artifact_path("artifacts/run_1/logs.txt", "metrics")
            Path("artifacts/run_1/metrics.jsonl")
        """
        parent = Path(run_log_path).parent

        # Map artifact types to filenames
        filenames = {
            "metrics": "metrics.jsonl",
            "checkpoints": "checkpoints.jsonl",
            "logs": "logs.log",
        }

        if artifact_type not in filenames:
            logger.error(f"Unknown artifact type: {artifact_type}")
            return None

        filename = filenames[artifact_type]

        # Try real cookbook location first (logs/ subdirectory)
        real_path = parent / "logs" / filename
        if real_path.exists():
            logger.debug(f"Found {artifact_type} at real path: {real_path}")
            return real_path

        # Fallback to simulation location (parent directory)
        fallback_path = parent / filename
        if fallback_path.exists():
            logger.debug(f"Found {artifact_type} at fallback path: {fallback_path}")
            return fallback_path

        logger.debug(
            f"Artifact not found: {artifact_type} for run at {run_log_path}. "
            f"Tried: {real_path}, {fallback_path}"
        )
        return None

    @staticmethod
    def get_metrics_path(run_log_path: str | Path) -> Optional[Path]:
        """Get path to metrics.jsonl file.

        Args:
            run_log_path: Path to run's log file

        Returns:
            Path to metrics.jsonl if exists, None otherwise
        """
        return ArtifactPathResolver.get_artifact_path(run_log_path, "metrics")

    @staticmethod
    def get_checkpoints_path(run_log_path: str | Path) -> Optional[Path]:
        """Get path to checkpoints.jsonl file.

        Args:
            run_log_path: Path to run's log file

        Returns:
            Path to checkpoints.jsonl if exists, None otherwise
        """
        return ArtifactPathResolver.get_artifact_path(run_log_path, "checkpoints")

    @staticmethod
    def get_training_logs_path(run_log_path: str | Path) -> Optional[Path]:
        """Get path to training logs.log file (detailed cookbook logs).

        Args:
            run_log_path: Path to run's log file

        Returns:
            Path to logs.log if exists, None otherwise
        """
        return ArtifactPathResolver.get_artifact_path(run_log_path, "logs")


def ensure_directory_exists(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
        return dir_path
    except OSError as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        raise


def safe_read_file(
    file_path: str | Path,
    encoding: str = "utf-8",
    max_size_mb: int = 100,
) -> Optional[str]:
    """Safely read a file with size limits and error handling.

    Args:
        file_path: Path to file to read
        encoding: Text encoding (default: utf-8)
        max_size_mb: Maximum file size in MB (default: 100)

    Returns:
        File contents as string, or None if error occurs
    """
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return None

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.error(
            f"File too large: {path} ({size_mb:.2f} MB > {max_size_mb} MB limit)"
        )
        return None

    try:
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        logger.debug(f"Successfully read file: {path} ({size_mb:.2f} MB)")
        return content
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {path}: {e}")
        return None
    except OSError as e:
        logger.error(f"OS error reading {path}: {e}")
        return None


def read_file_tail(
    file_path: str | Path,
    num_lines: int = 200,
    encoding: str = "utf-8",
) -> Optional[str]:
    """Read the last N lines of a file efficiently.

    Args:
        file_path: Path to file to read
        num_lines: Number of lines to read from end (default: 200)
        encoding: Text encoding (default: utf-8)

    Returns:
        Last N lines as string, or None if error occurs
    """
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return None

    try:
        from collections import deque

        with open(path, "r", encoding=encoding) as f:
            # Use deque with maxlen for memory-efficient tail reading
            lines = deque(f, maxlen=num_lines)

        result = "".join(lines)
        logger.debug(f"Read {len(lines)} lines from tail of {path}")
        return result
    except OSError as e:
        logger.error(f"Error reading file tail {path}: {e}")
        return None
