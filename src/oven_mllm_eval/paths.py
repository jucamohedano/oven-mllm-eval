# ==============================================================================
# Adapted from vlm-eval/src/vlmeval/paths.py
#
# Project-relative paths overridable via environment variables.
# ==============================================================================

import os
from pathlib import Path

from oven_mllm_eval import PACKAGE_DIR, PROJECT_ROOT, DATA_DIR

# ---------------------------------------------------------------------------
# Base data directory — override with VLMEVAL_DATA if you need a different
# layout on the cluster.
# ---------------------------------------------------------------------------
_data_root = Path(os.environ.get("VLMEVAL_DATA", str(DATA_DIR)))

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
INAT_IMAGES_DIR = _data_root / "inat_images"
OVEN_IMAGES_DIR = _data_root / "images/oven_images"

# ---------------------------------------------------------------------------
# OVEN processed data
# ---------------------------------------------------------------------------

# Prebuilt chains (already in the repo)
out_labels = str(DATA_DIR / "raw" / "oven_wikidata_chains_cleaned_labels.jsonl")

# Alias file (download separately)
OVEN_AKAS = str(DATA_DIR / "raw" / "wikidb_aka_oven_sample.jsonl")

# Bridged annotation file (produced by scripts/prepare_oven.py)
OVEN_SAMPLES = str(DATA_DIR / "processed" / "vlm_compatible_val.jsonl")

# ---------------------------------------------------------------------------
# Raw OVEN downloads (produced by scripts/prepare_oven.py)
# ---------------------------------------------------------------------------
VALIDATION_ENTITIES = str(DATA_DIR / "raw" / "oven_entity_val.jsonl")
OVEN_ID2_PATH = str(DATA_DIR / "raw" / "ovenid2impath.csv")

# ---------------------------------------------------------------------------
# Precomputed taxonomy index (produced by scripts/build_taxonomy_index.py)
# ---------------------------------------------------------------------------
OVEN_TAXONOMY_INDEX = str(DATA_DIR / "processed" / "oven_taxonomy_index.json")

# ---------------------------------------------------------------------------
# Stubs for files referenced by load_data.py originals but not needed at runtime.
# ---------------------------------------------------------------------------
chain_file = str(DATA_DIR / "raw" / "oven_wikidata_chains_raw.jsonl")
out_descs = str(DATA_DIR / "raw" / "oven_wikidata_chains_cleaned_descs.jsonl")
WIKI_FILE = str(DATA_DIR / "raw" / "Wiki6M_ver_1_0.jsonl")
oven_test_samples = str(DATA_DIR / "raw" / "oven_train_and_test_equal_repr_label_variants.jsonl")
wiki_id_file = str(DATA_DIR / "raw" / "oven_wikidata_ids.txt")
oven_only_target_test = str(DATA_DIR / "raw" / "oven_wikidata_chains_raw_only_test.jsonl")
oven_target_all = str(DATA_DIR / "raw" / "oven_wikidata_chains_raw.jsonl")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
MODEL_OUTPUT = str(PROJECT_ROOT / "results" / "generations")
MEASURE_SCORE_DIR = str(PROJECT_ROOT / "results" / "scores")
ESTIMATE_DIR = str(PROJECT_ROOT / "results" / "estimates")
CLIP_FOLDER = str(DATA_DIR / "clip")
