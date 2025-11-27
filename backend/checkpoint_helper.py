"""Checkpoint registration utilities for training runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from models import Run

from models import Checkpoint
from utils.file_utils import ArtifactPathResolver
from utils.json_utils import read_jsonl_file

logger = logging.getLogger(__name__)


async def register_checkpoint_from_logs(
    session: Session,
    run: Run,
    logs_path: Path,
) -> int:
    """Read checkpoints.jsonl and register checkpoints in database.

    Args:
        session: Database session
        run: Run model instance
        logs_path: Path to run's log file

    Returns:
        Number of checkpoints registered

    Raises:
        Exception: If checkpoint registration fails critically
    """
    # Use utility to get checkpoints file path with fallback logic
    checkpoints_file = ArtifactPathResolver.get_checkpoints_path(logs_path)

    if not checkpoints_file:
        logger.warning(f"[RUN {run.id}] No checkpoints.jsonl file found")
        return 0

    logger.info(f"[RUN {run.id}] Reading checkpoints from: {checkpoints_file}")

    try:
        # Use utility to read JSONL file
        checkpoints_data = read_jsonl_file(str(checkpoints_file), skip_errors=True)
        registered_count = 0

        for checkpoint_data in checkpoints_data:
            if "sampler_path" not in checkpoint_data:
                logger.warning(
                    f"[RUN {run.id}] Skipping checkpoint without sampler_path: {checkpoint_data}"
                )
                continue

            # Check if checkpoint already exists to avoid duplicates
            existing = (
                session.query(Checkpoint)
                .filter(
                    Checkpoint.run_id == run.id,
                    Checkpoint.tinker_path == checkpoint_data["sampler_path"],
                )
                .first()
            )

            if existing:
                logger.debug(
                    f"[RUN {run.id}] Checkpoint already exists: {checkpoint_data['sampler_path']}"
                )
                continue

            # Create checkpoint record in database
            checkpoint = Checkpoint(
                run_id=run.id,
                tinker_path=checkpoint_data["sampler_path"],
                kind="sampler",
                step=checkpoint_data.get("step", checkpoint_data.get("batch", 0)),
                meta=checkpoint_data,
            )
            session.add(checkpoint)
            registered_count += 1
            logger.info(
                f"[RUN {run.id}] Registered checkpoint: {checkpoint_data['sampler_path']}"
            )

        session.commit()
        logger.info(
            f"[RUN {run.id}] Successfully registered {registered_count} checkpoint(s)"
        )
        return registered_count

    except FileNotFoundError:
        logger.error(f"[RUN {run.id}] Checkpoints file not found: {checkpoints_file}")
        return 0
    except Exception as e:
        logger.error(
            f"[RUN {run.id}] Failed to register checkpoints: {e}", exc_info=True
        )
        session.rollback()
        raise
