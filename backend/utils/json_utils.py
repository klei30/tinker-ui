"""JSON utility functions with special handling for training metrics."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def parse_json_with_nan(json_string: str) -> Optional[dict[str, Any]]:
    """Parse JSON string and handle NaN values.

    Tinker cookbook metrics often contain NaN values for test metrics
    during training. This function parses JSON and converts NaN to None
    for proper JSON serialization.

    Args:
        json_string: JSON string to parse

    Returns:
        Parsed dictionary with NaN values converted to None,
        or None if parsing fails

    Examples:
        >>> parse_json_with_nan('{"loss": 1.5, "test_loss": NaN}')
        {'loss': 1.5, 'test_loss': None}

        >>> parse_json_with_nan('{"step": 1, "value": 1.234}')
        {'step': 1, 'value': 1.234}
    """
    try:
        data = json.loads(json_string)

        # Handle NaN values
        if isinstance(data, dict):
            _replace_nan_recursive(data)

        return data
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None


def _replace_nan_recursive(obj: Any) -> None:
    """Recursively replace NaN values with None in a data structure.

    Args:
        obj: Object to process (dict, list, or primitive)
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, float) and value != value:  # NaN check
                obj[key] = None
            elif value == "NaN":  # String NaN
                obj[key] = None
            elif isinstance(value, (dict, list)):
                _replace_nan_recursive(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, float) and item != item:  # NaN check
                obj[i] = None
            elif item == "NaN":
                obj[i] = None
            elif isinstance(item, (dict, list)):
                _replace_nan_recursive(item)


def read_jsonl_file(
    file_path: str,
    skip_errors: bool = True,
) -> list[dict[str, Any]]:
    """Read a JSONL (JSON Lines) file and parse all entries.

    Args:
        file_path: Path to JSONL file
        skip_errors: If True, skip malformed lines; if False, raise on error

    Returns:
        List of parsed JSON objects

    Raises:
        json.JSONDecodeError: If skip_errors=False and a line fails to parse
        FileNotFoundError: If file doesn't exist
    """
    results = []
    errors = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = parse_json_with_nan(line)
                    if data is not None:
                        results.append(data)
                    else:
                        errors += 1
                        if not skip_errors:
                            raise json.JSONDecodeError(
                                f"Failed to parse line {line_num}",
                                line,
                                0,
                            )
                except json.JSONDecodeError as e:
                    errors += 1
                    if skip_errors:
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num}: {e}"
                        )
                    else:
                        raise

        if errors > 0:
            logger.info(
                f"Read {len(results)} entries from {file_path} "
                f"({errors} errors skipped)"
            )
        else:
            logger.debug(f"Read {len(results)} entries from {file_path}")

        return results
    except FileNotFoundError:
        logger.error(f"JSONL file not found: {file_path}")
        raise


def safe_json_loads(
    json_string: str,
    default: Any = None,
) -> Any:
    """Safely parse JSON with a default fallback value.

    Args:
        json_string: JSON string to parse
        default: Value to return if parsing fails (default: None)

    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse failed, returning default: {e}")
        return default


def safe_json_dumps(
    obj: Any,
    indent: Optional[int] = None,
    default: Any = None,
) -> str:
    """Safely serialize object to JSON string.

    Args:
        obj: Object to serialize
        indent: Indentation level (None for compact)
        default: Default value to return if serialization fails

    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, indent=indent)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed: {e}")
        return default if default is not None else "{}"
