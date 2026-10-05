from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/pet_breed_classification/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"


class Config(BaseSettings):
    """Application settings with optional ``PETBREED_`` environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="PETBREED_",
        frozen=True,
        extra="ignore",
    )

    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    IMAGES_DIR: Path = RAW_DIR / "images"
    ANNOTATIONS_DIR: Path = RAW_DIR / "annotations"
    LABEL_MAP_PATH: Path = DATA_ROOT / "label_map.json"
    SPLIT_INDEX_PATH: Path = DATA_ROOT / "split_index.json"
    CORRUPTED_DIR: Path = DATA_ROOT / "corrupted"
    MANIFEST_PATH: Path = DATA_ROOT / "processed" / "manifest.json"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODEL_PATH: Path = ARTIFACTS_DIR / "resnet50_best.pt"
    ONNX_PATH: Path = ARTIFACTS_DIR / "model.onnx"
    MODEL_META_PATH: Path = ARTIFACTS_DIR / "model_meta.json"
    MODEL_VERSION: str = "v1"
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_BATCH_FILES: int = 32
    LOW_RES_WARN_SIDE: int = 64
    LOG_LEVEL: str = "INFO"

    # Reproducibility
    SEED: int = 42

    # Split: test = official test.txt; val is carved from trainval.txt (stratified by breed)
    VAL_RATIO: float = 0.20

    # Corruption suite
    # None = corrupt the full test split; an int = sample that many images per class
    CORRUPTION_SAMPLE_PER_CLASS: int | None = None

    # Parallelism: None = use all CPU cores minus one
    NUM_WORKERS: int | None = 10

    # Training
    EPOCHS: int = 10
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4


cfg = Config()
