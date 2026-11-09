"""3-Channel Physics-Motivated preprocessing experiment.

This package is a SELF-CONTAINED experiment. It does not modify the baseline
`space_debris` pipeline; it only reuses read-only utilities from it
(class taxonomy, Google Drive sync, YOLO model builders).

Modules:
    physics_pipeline   - the 3-channel preprocessing function itself
    dataset_processor  - dataset pooling, 80/20 re-split, batch processing, labels
    drive_upload       - automatic upload of processed_dataset/ to Google Drive
    visualization      - 6-panel sample comparisons + channel statistics
    validation         - dataset integrity / channel-independence checks
    experiment_config  - loader for configs/physics_config.yaml
"""

from .physics_pipeline import (
    PhysicsPipelineConfig,
    DEFAULT_PIPELINE_CONFIG,
    preprocess_image_3ch,
)

__all__ = [
    "PhysicsPipelineConfig",
    "DEFAULT_PIPELINE_CONFIG",
    "preprocess_image_3ch",
]
