# Predictive Maintenance for Industrial Equipment

Failure prediction and Remaining Useful Life estimation on the NASA C-MAPSS
turbofan engine dataset, with a live Streamlit dashboard.

Classical ML answers *"will this engine fail soon"*, an LSTM answers
*"how many cycles remain"*. Everything runs locally, no cloud services.

**B.Tech 7th Semester Final Year Project**
Priyadarshini College of Engineering, Nagpur
Team: Himanshu Gajbhiye · Atharva Awachat · Mayank Guneria · Sujal Dube
Guide: Prof. H. M. Kubade

---

## Quick start

Three commands from a clean clone:

```powershell
.\setup.ps1                        # Windows
python -m src.train_classifier     # trains RF + XGBoost, ~40 seconds
streamlit run dashboard/app.py     # opens the dashboard
```

On macOS or Linux use `bash setup.sh` for the first line.

The dataset is committed to this repo, so there is nothing to download.

---

## Setup

### Option A — automated script (recommended)

**Windows PowerShell**

```powershell
git clone https://github.com/aarshnomorecool/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
git clone https://github.com/aarshnomorecool/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
bash setup.sh
source .venv/bin/activate
```

The script creates `.venv`, installs everything in `requirements.txt`, and runs
the 14-check verification suite so you know the data layer is sound before
training anything. Re-running it is safe; an existing `.venv` is reused.

If PowerShell blocks the script, allow it for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Option B — let an AI agent set it up

Paste this into Antigravity, Codex, Cursor, or any coding agent with shell
access:

```
Clone https://github.com/aarshnomorecool/predictive-maintenance-cmapss.git and set it up.

1. Create a Python 3.10+ virtual environment at .venv in the project root
2. Install everything from requirements.txt into it
3. Run `python verify_foundation.py` and confirm all 14 checks pass
4. Run `python -m src.train_classifier` and report the final ROC-AUC and recall
5. Tell me the command to launch the dashboard, but do not launch it yourself

The dataset is already in the repo under dataset-abstract/, do not download
anything. Everything runs locally, do not add cloud or Colab dependencies.
The design rules the code must obey are documented in the README section below.
```

### Option C — manual

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
python verify_foundation.py
```

Requires **Python 3.10 or newer**. Tested on 3.13.

---

## Commands

| Command | What it does |
|---|---|
| `python -m src.data_pipeline` | Load, label and split FD001, print a summary |
| `python -m src.preprocessing` | Scale, balance and window, print array shapes |
| `python -m src.eda` | Write figures to `reports/figures/` and `reports/eda_summary.md` |
| `python verify_foundation.py` | 14 correctness checks over the data foundation |
| `python -m src.train_classifier` | **Train RF + XGBoost**, live progress, saves to `models/` |
| `streamlit run dashboard/app.py` | **Launch the dashboard** |

### Training options

```bash
python -m src.train_classifier                        # default: 300 trees, 500 rounds
python -m src.train_classifier --quick                # ~3s smoke run, scores not meaningful
python -m src.train_classifier --no-smote             # train on the real 15/85 class prior
python -m src.train_classifier --trees 500 --rounds 1000
```

Training prints live validation ROC-AUC as it goes. Every number that scrolls
past is a real score from a real partially-trained model, not a progress
animation. Expect roughly 40 seconds for the default run.

### Dashboard

Train before launching. If no models exist the dashboard says so and prints the
command rather than erroring out. Retraining while the dashboard is open is
picked up automatically within about 5 seconds, no restart needed.

---

## Results

FD001 held-out test set, 100 engines, threshold 0.5, all 13,096 rows:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9834 | 0.6542 | 0.7349 | 0.6922 | **0.9906** |
| XGBoost | 0.9806 | 0.5882 | 0.7831 | 0.6718 | **0.9903** |

**Accuracy is inflated and should not be reported as the headline.** Only 2.54%
of test rows are positive, because test trajectories stop well before failure.
A model that predicts "healthy" for everything scores 97%. Recall and ROC-AUC
are the meaningful columns.

Evaluated per engine at its latest cycle (the operationally realistic question,
which is what the dashboard's Fleet tab shows), XGBoost at a 0.40 threshold
flags 26 engines, 25 are genuinely failing, and 22 are caught: **88% recall,
85% precision**.

---

## Dataset

NASA C-MAPSS turbofan engine degradation simulation. Committed under
`dataset-abstract/`, so a clone runs immediately.

| Subset | Train rows | Test rows | Conditions | Fault modes | Status |
|---|---|---|---|---|---|
| **FD001** | 20,631 | 13,096 | 1 | 1 | **in use** |
| FD002 | 53,759 | 33,991 | 6 | 1 | needs per-regime normalization |
| FD003 | 24,720 | 16,596 | 1 | 2 | easiest extension |
| FD004 | 61,249 | 41,214 | 6 | 2 | needs per-regime normalization |

26 space-separated columns, no header: unit number, cycle, 3 operational
settings, 21 sensor measurements.

FD002 and FD004 have six operating conditions, so sensor values shift with the
regime and need per-regime normalization before a single scaler is valid. FD003
shares FD001's single condition and would need no new preprocessing.

Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008. Paper
included under `non-huma-PDF-format/`.

---

## How it works

### Two branches, deliberately separate

| | Classification | Regression |
|---|---|---|
| Question | Will this engine fail soon? | How many cycles remain? |
| Target | `failure_soon` = RUL ≤ 30 | clipped `RUL` |
| Shape | one row per cycle, 17 features | windows of 30 cycles × 17 features |
| Models | Random Forest, XGBoost | LSTM |
| SMOTE | training split only | never |

### Design rules the code enforces

**RUL is clipped at 125.** Sensors are flat early in an engine's life and
degradation is not yet observable, so an unbounded target asks the model to
distinguish 300 cycles remaining from 250 using noise. Piecewise-linear RUL is
standard practice on this dataset.

**Test RUL comes from the ground-truth vector.** Test engines stop before
failure, so RUL at any cycle is `(last_recorded_cycle - current_cycle)` plus the
value `RUL_FD001.txt` gives for that engine's final cycle.

**Near-constant columns are dropped by measured variance, not a hardcoded
list.** On FD001 this removes `setting_3` and sensors 1, 5, 10, 16, 18 and 19,
leaving 17 features. Note that `sensor_6` is *kept*: it varies between 21.60 and
21.61, so it is not literally constant even though it appears on the usual
textbook list of flat sensors.

**Train and validation split by engine unit, never by row.** Random row
splitting puts later cycles of an engine into training and earlier cycles of the
same engine into validation, leaking the future and inflating every metric. This
is the single most common mistake in predictive maintenance projects.

**The scaler is fitted on the training split alone.** Fitting on the full frame
lets test minima and maxima bleed into the transform. It is a leak even though
no label is involved.

**SMOTE touches the training split only.** Balancing validation or test data
changes the class prior the metrics are measured against and makes precision and
recall meaningless.

**LSTM windows never span two engines.** A window that crosses the boundary
splices one engine's failure onto another's healthy start.

**Inference reuses the saved scaler.** The dashboard loads
`models/scaler.joblib` rather than refitting. A refit would usually match, but
"usually" is how a transform quietly drifts from the one the model trained on.

`verify_foundation.py` asserts every one of the above against the real data.
Each check guards a mistake that would silently inflate metrics rather than
raise an error.

---

## Dashboard

Four tabs:

- **Fleet** — every engine ranked by failure probability, with how many alerts
  the model raised, how many engines are genuinely in the failure window, and
  how many were caught
- **Engine N** — one engine's probability curve across its whole life, with the
  alert threshold and the true failure onset both marked, plus raw sensor traces
- **Model performance** — metrics table and feature importance
- **Roadmap** — what is not built yet, stated plainly rather than mocked up

Sidebar controls the model, the alert threshold, the selected engine, and
whether the page auto-reloads when `models/` changes.

---

## Project status

**Built**

- Scaffolding, config, data pipeline, preprocessing
- EDA with six figures and a written summary
- 14-check verification suite
- Random Forest and XGBoost classifiers with live training output
- Inference layer shared by the dashboard and the planned Flask API
- Streamlit dashboard

**Not built yet**

- `src/train_lstm.py` — RUL regression. The windowing that feeds it is already
  built and tested: 14,459 training sequences of shape (30, 17)
- `src/explainability.py` — SHAP TreeExplainer
- `src/alerts.py` — email alerts via `smtplib`
- FD002 / FD004 extension with per-regime normalization

### Known limitations

Stated openly rather than discovered during a demo:

- The dashboard's alert-threshold slider spans 0.05 to 0.95, but XGBoost only
  outputs roughly 0.06 to 0.93. At the extreme ends the slider degenerates:
  0.05 flags all 100 engines, 0.95 flags none
- The Fleet tab's "True RUL" column shows the *clipped* value. Eleven engines
  display 125 when their real remaining life is 126 to 145 cycles
- Random Forest probabilities are bimodal (34 of 100 engines sit at exactly
  0.000 or 1.000), so the threshold slider feels coarse on RF and smooth on
  XGBoost
- Auto-refresh on retrain is implemented and its file-signature logic is
  verified, but the browser reload path has not been tested end to end
- Training writes a 53 MB model file non-atomically, so a dashboard refresh
  landing mid-write could read a truncated file

---

## Layout

```
src/
  config.py            paths, columns, hyperparameters
  data_pipeline.py     loading, RUL labelling, feature pruning, unit split
  preprocessing.py     scaling, SMOTE, windowing
  eda.py               figures and written summary
  train_classifier.py  RF + XGBoost with live training output
  inference.py         artifact loading and scoring, shared by UI and future API
dashboard/
  app.py               Streamlit UI
models/                trained artifacts, gitignored, regenerate by training
reports/
  eda_summary.md
  figures/
dataset-abstract/      C-MAPSS text files
non-huma-PDF-format/   Saxena et al. PHM 2008 paper
verify_foundation.py   14 correctness checks
setup.ps1 / setup.sh   one-shot environment setup
```

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn ·
TensorFlow/Keras · Matplotlib · seaborn · Altair · Streamlit · SHAP · Flask

Everything runs on local CPU. No Colab, no cloud storage, no hosted training.
