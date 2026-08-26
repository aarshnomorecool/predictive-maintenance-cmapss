"""Central configuration for the predictive maintenance pipeline.

Every path, column name and hyperparameter that more than one module needs
lives here so nothing gets hardcoded twice and drifts apart.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths -----

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _locate_data_dir() -> Path:
    """Find the folder holding train_FD001.txt.

    The dataset zip is meant to extract at the project root, but the copy
    currently checked in sits under dataset-abstract/. Look in both so the
    pipeline works either way instead of failing on a path nobody updated.
    """
    candidates = [PROJECT_ROOT]
    candidates += sorted(PROJECT_ROOT.glob("*/"))
    candidates += sorted(PROJECT_ROOT.glob("*/*/"))
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "train_FD001.txt").exists():
            return candidate
    raise FileNotFoundError(
        "train_FD001.txt not found. Extract the C-MAPSS dataset into the "
        f"project root ({PROJECT_ROOT}) and try again."
    )


DATA_DIR = _locate_data_dir()

# --------------------------------------------------------------- columns ----

UNIT_COL = "unit"
CYCLE_COL = "cycle"
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = [UNIT_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS

RUL_COL = "RUL"
LABEL_COL = "failure_soon"

# ------------------------------------------------------------- behaviour ----

SUBSET = "FD001"          # start here; FD002/FD004 need per-regime scaling
RANDOM_SEED = 42

# Rule 1: piecewise-linear RUL. Degradation is not observable before this many
# cycles remain, so a flat ceiling is a truer target than an unbounded ramp.
RUL_CLIP = 125

# Binary classification target: an engine is "failing soon" once RUL drops to
# this many cycles or fewer. 30 is the common C-MAPSS convention.
FAILURE_THRESHOLD = 30

# Rule 5: sensors whose normalised spread falls below this are near-constant
# and carry no signal. Measured, not hardcoded.
NEAR_CONSTANT_STD_THRESHOLD = 1e-6

# Rule 4: LSTM sequence length. A window never spans two engine units.
WINDOW_SIZE = 30

# Fraction of *engine units* (rule 3, never rows) held out for validation.
VAL_UNIT_FRACTION = 0.2
