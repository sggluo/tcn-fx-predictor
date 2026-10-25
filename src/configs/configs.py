from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR/'data'
DATA_DIR_RAW= DATA_DIR/'raw'
DATA_DIR_PROCESSED= DATA_DIR/'processed'
DATA_DIR_FEATURES_TARGETS= DATA_DIR/'features_targets'
