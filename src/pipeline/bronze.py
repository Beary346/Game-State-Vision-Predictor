"""Bronze layer — re-export shim.

The bronze implementation lives in ``src.ingestor_bronze`` (frame extraction /
bronze team, per AGENTS.md). This module re-exports its public API so existing
``from src.pipeline.bronze import ...`` imports keep working while the code
stays in a single source of truth.
"""

from src.ingestor_bronze import (
    DEFAULT_FRAME_INTERVAL,
    MAX_VOD_DURATION_SEC,
    VIDEO_EXTENSIONS,
    BronzeFeatures,
    H,
    VodMetadata,
    W,
    denoise,
    enhance_contrast,
    extract_frames,
    extract_metadata,
    get_vod_info,
    load_image,
    preprocess,
    process_image,
    process_input,
    process_vod,
    save_image,
    sharpen,
    validate_vod_duration,
)

__all__ = [
    "DEFAULT_FRAME_INTERVAL",
    "MAX_VOD_DURATION_SEC",
    "VIDEO_EXTENSIONS",
    "BronzeFeatures",
    "H",
    "VodMetadata",
    "W",
    "denoise",
    "enhance_contrast",
    "extract_frames",
    "extract_metadata",
    "get_vod_info",
    "load_image",
    "preprocess",
    "process_image",
    "process_input",
    "process_vod",
    "save_image",
    "sharpen",
    "validate_vod_duration",
]


if __name__ == "__main__":
    """Make ``python -m src.pipeline.bronze`` behave like the real CLI.

    The shim is a re-export module and carries no its own argparse; we re-run
    the canonical ingestor module as ``__main__`` so the documented command in
    ``docs/data_collection_plan.md`` works verbatim.
    """

    import runpy

    runpy.run_module("src.ingestor_bronze", run_name="__main__")
