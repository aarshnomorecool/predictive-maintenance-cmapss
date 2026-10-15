# Predictive Maintenance for Industrial Equipment

Failure prediction and Remaining Useful Life estimation on the NASA C-MAPSS
turbofan engine dataset, with a live web dashboard built around an interactive
3D cutaway of the engine.

Classical ML answers *"will this engine fail soon"*, an LSTM answers
*"how many cycles remain"*. Everything runs locally, no cloud services.

**B.Tech 7th Semester Final Year Project**
Priyadarshini College of Engineering, Nagpur
Team: Himanshu Gajbhiye · Atharva Awachat · Mayank Guneria · Sujal Dube
Guide: Prof. H. M. Kubade

---

## Quick start

Four commands from a clean clone:

```powershell
.\setup.ps1                        # Windows
python -m src.train_classifier     # trains RF + XGBoost, ~40 seconds
python -m src.train_lstm           # trains the remaining-life LSTM, ~1.5 minutes
python -m api.server               # then open http://127.0.0.1:8050
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
5. Run `python -m src.train_lstm` and report the final test RMSE and NASA score
6. Tell me the command to launch the dashboard (`python -m api.server`), but do not launch it yourself

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
| `python -m src.train_lstm` | **Train the remaining-life LSTM**, live progress, saves to `models/` |
| `python -m src.explainability` | SHAP attribution: which sensors drove a prediction |
| `python -m src.latent_space` | Checks the LSTM's vector space is meaningful: neighbour-based life estimate vs LSTM vs a constant guess |
| `python -m src.simulator` | Checks the trained models read simulated engines the way they read real ones |
| `python -m src.alerts --test` | Shows the email settings and sends one test email |
| `python -m src.sensor_health` | Per-sensor health calibration, and a check that the colour bands separate failing engines |
| `python -m api.server` | **Launch the dashboard** at http://127.0.0.1:8050 |
| `streamlit run dashboard/app.py` | The earlier, simpler Streamlit view |

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

```bash
python -m src.train_lstm               # default: up to 80 epochs, early stopping
python -m src.train_lstm --quick       # ~5s smoke run, scores not meaningful
python -m src.train_lstm --epochs 150 --patience 20
```

The LSTM prints validation RMSE in cycles after every epoch and keeps the
weights from the best epoch. CPU only (TensorFlow has no GPU support on native
Windows), about 2 seconds per epoch.

### Explaining a prediction

```bash
python -m src.explainability                       # highest-risk engine, XGBoost
python -m src.explainability --engine 24           # a specific engine
python -m src.explainability --model RandomForest  # slower, ~0.15s per row
```

Prints fleet-wide SHAP importance and then a per-sensor breakdown of one
engine's score, with the contributions summing back to the model's actual
output.

### Dashboard

```bash
python -m api.server
```

Then open http://127.0.0.1:8050. Train the classifiers before launching. If no
models exist the page says so and shows the command, then fills itself in once
training finishes. The LSTM is optional: without it the dashboard works and
the remaining-life panels say how to train it. Retraining either while the page
is open is picked up within about 5 seconds, no restart or reload needed.

Everything is served locally, including Three.js (vendored under `web/vendor/`),
so the dashboard works with no internet connection.

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
which is what the dashboard's fleet list shows), XGBoost at a 0.40 threshold
flags 26 engines, 25 are genuinely failing, and 22 are caught: **88% recall,
85% precision**.

### Remaining useful life (LSTM)

FD001 test set, 100 engines, predicted RUL at each engine's last recorded cycle:

| Model | RMSE (cycles) | MAE (cycles) | NASA score |
|---|---|---|---|
| LSTM | **13.58** | **10.61** | **288.0** |
| Average-life guess | 40.69 | 35.00 | 19,887.2 |

The average-life guess predicts the same number (the mean training RUL) for
every engine. Beating it by a factor of three on RMSE shows the LSTM is reading
the sensors, not the base rate. Published LSTM results on FD001 typically fall
between 13 and 16 RMSE.

The **NASA score** is the official PHM 2008 challenge metric (Saxena et al.,
eq. 1). It penalises late predictions exponentially harder than early ones,
because predicting failure too late means an engine failing in service. Lower
is better.

True RUL is capped at 125 for scoring, the same cap used in training and the
convention in most published FD001 work. The model therefore cannot tell 125
cycles left from 200: early in a run it predicts just under 125, then tracks
the true value once degradation shows in the sensors. The dashboard's
remaining-life chart shows exactly this.

Training is seeded, but TensorFlow on CPU is not guaranteed bit-for-bit
deterministic, so a retrain can land slightly off these exact numbers.

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

**SHAP contributions are read in the model's own units.** Random Forest
attribution is in probability, XGBoost's is in log-odds and needs a sigmoid to
become the number on screen. The two are not comparable side by side, and the
code labels which is which rather than quietly mixing them.

**Inference reuses the saved scaler.** The dashboard loads
`models/scaler.joblib` rather than refitting. A refit would usually match, but
"usually" is how a transform quietly drifts from the one the model trained on.

`verify_foundation.py` asserts every one of the above against the real data.
Each check guards a mistake that would silently inflate metrics rather than
raise an error.

---

## Dashboard

A Flask backend (`api/server.py`) serves a JSON API over the same inference,
SHAP and sensor-health code the CLI tools use, plus a static frontend in `web/`.
Nothing on screen is mocked: every number comes from the trained models and the
C-MAPSS test split.

**3D engine.** A cutaway of a two-spool high-bypass turbofan with every one of
the 21 C-MAPSS channels placed at its measurement station (fan, bypass duct,
LPC, HPC, combustor, HPT, LPT, nozzle), following Table 2 of Saxena et al. The
geometry is illustrative, not a scan of a real engine, and the page says so.
Drag to orbit, scroll to zoom, click a sensor to inspect it.

**Sensor colours.** Each sensor's reading is placed on a scale from its healthy
level to its failure level, both measured on training engines only:

| Colour | Meaning |
|---|---|
| Green | Reads like a healthy engine (under 45% of the way to failure level) |
| Amber | Drifting (45% to 75%) |
| Red | Near the level engines reach in their last 10 cycles (75% and above) |
| Grey | Too little signal: the sensor barely changes with wear (P15) |
| Hollow | Not used by the models, reading is constant across the fleet |

The bands are checked, not guessed: at their latest cycle, engines inside the
failure window average 5.6 red sensors, engines outside it average 0.2
(`python -m src.sensor_health` prints this).

**Clicking a sensor** draws a leader-line callout on the engine with a severity
bar, and the side panel shows the reading over the engine's whole run against
its healthy and failure levels, its drift towards failure, and how much it
moved the model's score at that cycle (SHAP).

**Three tabs, one engine.** The centre panel switches between three views of
the selected engine, all driven by the same replay slider:

- **Engine** — the 3D cutaway with live sensor colours, described above
- **Vector space** — a 3D point cloud of the LSTM's internal state. Each of
  the 20,631 dots is one cycle of one training engine, placed by the 32
  numbers the LSTM's second layer uses to summarise the 30 cycles before it,
  projected to 3D with PCA (97.2% of the variance kept), and coloured by how
  many cycles that engine really had left. Healthy engines sit at the green
  end, failing ones at the red end, and the selected test engine draws its
  path between them. Blue lines join it to its 8 nearest training
  neighbours, and the side panel lists how long each of them lasted
- **Neural network** — the trained LSTM drawn layer by layer (input window,
  LSTM 64, LSTM 32, Dense 16, output), with lines weighted by the real
  trained connection strengths and every neuron coloured by the value it
  actually computed for this engine and cycle. Changing cycle sends a pulse
  through the network from input to prediction

The vector space is checked, not just drawn: averaging the remaining life of
a test engine's 8 nearest training neighbours predicts it with RMSE 15.96
cycles, against 13.58 for the LSTM and 40.69 for guessing the average
(`python -m src.latent_space`). Neighbour search is exact. A vector database
uses an approximate index because it holds millions of vectors; at 20,631 an
exact search answers in milliseconds.

**Live simulation.** The first row of the fleet list, **Live**, is a virtual
engine that generates a new cycle every second (adjustable 1× to 10×). All
three models score it as it runs: failure risk, predicted life left, sensor
colours, SHAP drivers, the vector space and the network view all update
live. Controls under the engine:

- **Play / pause**, **speed**, and **wear rate** (normal, fast, severe)
- **Inject HPC fault** — makes degradation progress three times faster.
  FD001 has one fault mode, HPC degradation, so this speeds up the fault the
  models were trained on rather than inventing one they never saw
- **New engine** — fits a fresh engine with its own random initial wear and
  manufacturing offsets
- **Run demo** — a fixed, repeatable scenario for presentations: healthy
  running, then an HPC fault at cycle 75, the alert at about cycle 89, and
  end of life at cycle 103. About a minute at 2×
- **Set a sensor by hand** — click any sensor on the 3D engine, then pick a
  value. The engine reads it (plus normal noise) every cycle until released.
  The slider shows the range training engines actually recorded, and warns
  when a value falls outside it, because the models' response there is not
  reliable

The simulator is built from statistics measured on the real training
engines, not invented numbers: every channel's healthy level, failure
level, noise within one engine, and spread between engines. One hidden
damage value drives all sensors together, growing exponentially as C-MAPSS
degradation does. Because the wear rate is known, the simulator also knows
the true remaining life, shown as "projected life left".

It is checked against the models (`python -m src.simulator`): under normal
wear the LSTM predicts simulated engines with RMSE 14.9 cycles, close to its
13.58 on real test engines, and both classifiers separate failing from
healthy cycles with ROC-AUC 0.94. Under fast wear or an injected fault the
LSTM is less accurate (RMSE 22 to 29) and alerts come later, because every
real FD001 engine degrades at a natural pace and the model learned that
timescale. That is a property of the trained model, shown honestly, not
hidden.

**Email alerts.** Alerts are raised for the live engine and sent by email.
The rule, chosen by measuring candidates on the 20 real validation engines
(which ran to failure and were never used to fit the classifiers):

> Alert when the 5-cycle average failure risk reaches the alert level, or the
> LSTM predicts 30 cycles or fewer.

| Rule | Shortest warning, real engines | Early false alarms | Shortest warning, HPC fault |
|---|---|---|---|
| Single cycle over 50% | 22 cycles | 1 | 7 cycles |
| **5-cycle average, or LSTM ≤ 30** | **27 cycles** | **0** | **15 cycles** |

Averaging stops one noisy reading from raising an alarm; the LSTM clause
catches fast faults the classifier reacts to late. Each engine alerts once,
then only again after it has clearly recovered, so a failing engine does not
send an email every cycle. The email gives the risk, predicted life, the
three sensors pushing the risk up most (SHAP) and a link to the engine on the
dashboard. The **Alerts** button in the top bar shows every alert, whether
its email was sent, and the exact text.

To turn on email, copy `.env.example` to `.env` and fill it in (Gmail needs
an [app password](https://myaccount.google.com/apppasswords), not your normal
password; `.env` is gitignored). No restart needed. Without it, alerts are
still raised and shown, marked "email not set up". College and venue
networks often block outgoing mail on port 587; if sending fails, the
dashboard says so, and a phone hotspot usually works.

**Replay.** The play control and cycle slider step through the engine's
recorded cycles. Sensor colours, risk, remaining life and SHAP drivers all
follow the selected cycle, so you can watch an engine degrade.

**Elsewhere on the page:** the fleet list ranked by risk with live alert counts,
a model switch (Random Forest / XGBoost), an alert slider bounded to the range
the selected model actually outputs, a model report with held-out metrics, and
a station strip under the engine that doubles as a keyboard-accessible way to
pick sensors. The URL names the engine and sensor on screen, so any view can be
bookmarked.

The earlier Streamlit view is still in `dashboard/app.py` for comparison.

---

## Project status

**Built**

- Scaffolding, config, data pipeline, preprocessing
- EDA with six figures and a written summary
- 14-check verification suite
- Random Forest and XGBoost classifiers with live training output
- LSTM remaining-life model (TensorFlow/Keras), scored with RMSE, MAE and the
  NASA PHM 2008 metric, shown live in the dashboard
- Atomic model saves, so a dashboard refresh never reads a half-written file
- Vector-space view of the LSTM's internal state with nearest-neighbour
  search, and a live neural-network diagram showing data flowing through
- Inference layer shared by the dashboard, the API and the CLI tools
- SHAP explanations, per prediction and fleet-wide
- Per-sensor health scoring, calibrated on training engines
- Flask API and web dashboard with the interactive 3D engine
- Streamlit dashboard (earlier, simpler view)

- Live engine simulator built from measured training statistics, with
  manual sensor overrides and a scripted demo
- Email alerts with a measured alert rule, cooldown, and an in-dashboard log

**Not built yet**

- FD002 / FD004 extension with per-regime normalization

### Known limitations

Stated openly rather than discovered during a demo:

- Random Forest probabilities are bimodal (34 of 100 engines sit at exactly
  0.000 or 1.000), so the alert slider feels coarse on RF and smooth on
  XGBoost
- The older Streamlit view still has a fixed 0.05 to 0.95 slider and shows
  clipped true RUL. The web dashboard fixes both: its slider is bounded to the
  selected model's real output range, and it shows unclipped remaining life
- 54 of the 100 LSTM test predictions are late (they overestimate remaining
  life), which is the costlier direction. Its NASA score is still low because
  the late errors are small, but a deployment would want a safety margin or an
  asymmetric loss that pushes predictions earlier
- The LSTM gives one number with no stated uncertainty. A prediction range
  (for example conformal intervals calibrated on the validation engines) is
  the natural next step
- Sensor colours measure drift towards failure, not a fault diagnosis. A red
  sensor means "reads like an engine near the end of its life", which is what
  the data supports, not "this component has broken"

---

## Layout

```
src/
  config.py            paths, columns, hyperparameters
  data_pipeline.py     loading, RUL labelling, feature pruning, unit split
  preprocessing.py     scaling, SMOTE, windowing
  eda.py               figures and written summary
  train_classifier.py  RF + XGBoost with live training output
  train_lstm.py        remaining-life LSTM with live training output
  artifact_io.py       atomic model saves, so the dashboard never reads half a file
  inference.py         artifact loading and scoring, shared by UI and future API
  explainability.py    SHAP attribution, per prediction and fleet-wide
  sensor_health.py     per-sensor drift, severity bands, sensor metadata
  latent_space.py      LSTM vectors, nearest-neighbour index, PCA, activations
  simulator.py         live virtual engine built from training statistics
  alerts.py            alert rule, email sending, alert log
api/
  server.py            Flask JSON API, serves the web dashboard
web/
  index.html           dashboard page
  js/engine3d.js       procedural 3D turbofan and sensor markers
  js/main.js           state, API calls, panels, replay, live sync
  js/charts.js         SVG charts
  js/vectorspace.js    3D point cloud of the LSTM's vector space
  js/network.js        neural-network diagram with data-flow animation
  vendor/              Three.js r170, so the page runs offline
dashboard/
  app.py               earlier Streamlit UI
models/                trained artifacts, gitignored, regenerate by training
reports/
  eda_summary.md
  figures/
dataset-abstract/      C-MAPSS text files
non-huma-PDF-format/   Saxena et al. PHM 2008 paper
.env.example           email alert settings template; copy to .env
verify_foundation.py   14 correctness checks
setup.ps1 / setup.sh   one-shot environment setup
```

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn ·
TensorFlow/Keras · Matplotlib · seaborn · Altair · Streamlit · SHAP · Flask ·
Three.js

Everything runs on local CPU. No Colab, no cloud storage, no hosted training.
