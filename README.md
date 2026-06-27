# movieRecommenderSystem

A Flask-based movie recommender that ships two interchangeable variants of
the same application for side-by-side A/B evaluation:

- **Original (baseline)** — the course demo: item-based kNN on raw ratings,
  rate-then-see-similar UI.
- **Enhanced (group project)** — hybrid time-decayed User-kNN + TF-IDF
  content model + MMR diversity re-ranking, wrapped in a richer UI with
  "Why?" explanations, a Diversity slider, a *Haven't seen / Interested /
  Not interested* onboarding flow, a *Trending Now* strip, a *Your Taste*
  analytics dashboard, and a *Profile Switcher* for multi-participant
  within-subject testing.

Both variants run from the same codebase; two environment variables decide
which algorithm and which template are active.

---

## Table of contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Running the app](#3-running-the-app)
4. [Project structure](#4-project-structure)
5. [Enhanced features](#5-enhanced-features)
6. [A/B test workflow](#6-ab-test-workflow)
7. [Evaluation pipeline](#7-evaluation-pipeline)
8. [Dataset](#8-dataset)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Requirements

| Item       | Version / Notes                                         |
|------------|---------------------------------------------------------|
| OS         | Windows 10/11 (commands below use Anaconda Prompt)      |
| Python     | 3.10 (recommended; 3.12 also tested)                    |
| Conda      | Anaconda or Miniconda                                   |
| Browser    | Chrome / Edge / Firefox (modern, ES modules required)   |
| Disk       | ~500 MB for dependencies + dataset                      |

Key Python dependencies (full list in `requirements.txt`):
`Flask 3.x`, `pandas`, `numpy`, `scikit-learn`, `scikit-surprise`,
`scipy`, `nltk`, `pyquery`, `lxml`.

---

## 2. Installation

Open **Anaconda Prompt** (not PowerShell / cmd) and run:

```cmd
:: 1. Create and activate the env
conda create -n lab3 python=3.10 -y
conda activate lab3

:: 2. cd to the project root
cd /d D:\path\to\Demo_materials

:: 3. Install build tools and scikit-surprise (needs conda-forge)
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise -y

:: 4. Install the rest
pip install -r requirements.txt
```

Verify the install:

```cmd
python -c "import flask, sklearn, surprise, pandas; print('OK')"
```

Expected output: `OK`.

---

## 3. Running the app

### 3.1 One-click scripts (recommended)

From the project root inside an active `lab3` Anaconda Prompt:

```cmd
scripts\run_enhanced.bat      :: Enhanced variant  (condition B)
scripts\run_original.bat      :: Original baseline (condition A)
```

Each script sets `REC_ALGO` + `UI_MODE` and calls
`flask --app flaskr run --debug`. Wait for the line
`* Running on http://127.0.0.1:5000`, then open
<http://127.0.0.1:5000> in a browser. The terminal window hosts the
server — closing it stops the app.

### 3.2 Manual start (environment variables)

Two env-vars select the variant; both default to `enhanced`.

| Variable   | Values                    | Effect                                              |
|------------|---------------------------|-----------------------------------------------------|
| `REC_ALGO` | `original` \| `enhanced`  | Recommendation pipeline (`flaskr/tools/recommenders.py`) |
| `UI_MODE`  | `original` \| `enhanced`  | Template: `index_original.html` vs `index.html`     |

Anaconda Prompt / `cmd.exe`:

```cmd
set REC_ALGO=enhanced
set UI_MODE=enhanced
flask --app flaskr run --debug
```

PowerShell:

```powershell
$env:REC_ALGO="original"; $env:UI_MODE="original"
flask --app flaskr run --debug
```

### 3.3 Stopping the server

Press **Ctrl+C** (sometimes twice — watchdog reloader spawns a child
process). If it refuses, press **Ctrl+Break** or close the window.
See [Troubleshooting](#9-troubleshooting) if port 5000 stays occupied.

---

## 4. Project structure

```
Demo_materials/
├── flaskr/                      Flask application package
│   ├── __init__.py              create_app() factory
│   ├── main.py                  Enhanced routes, hybrid scoring, /api/*
│   ├── main_original.py         Baseline routes (kept for reference)
│   ├── scrape.py                Poster / overview scraper
│   ├── static/
│   │   ├── app.js               Vue 3 front-end (Enhanced UI)
│   │   ├── img/                 Logos
│   │   └── ml_data/             MovieLens CSVs (ratings, movies, links)
│   ├── templates/
│   │   ├── index.html           Enhanced UI (Profile, Taste, Trending, ...)
│   │   ├── index_original.html  Original baseline UI
│   │   ├── _modals.html         Detail / Why / Taste / Interest modals
│   │   ├── _movie_card.html     Shared card partial
│   │   └── _onboarding.html     Genre + 10-movie rating flow
│   └── tools/
│       ├── recommenders.py      Time-decayed kNN + TF-IDF + MMR + explain
│       ├── data_tool.py         CSV loaders + caches
│       └── scrape_tool.py       TMDb scraping helpers
├── evaluation/                  A/B test artifacts (see §6, §7)
├── report/
│   ├── report_template.md       Report skeleton with TODO anchors
│   └── figures/                 Output directory for plots
├── scripts/
│   ├── run_enhanced.bat         Start Enhanced variant
│   ├── run_original.bat         Start Original baseline
│   └── smoke_test_recommenders.py  Offline sanity check
├── requirements.txt
├── LICENSE
└── README.md                    You are here
```

## 5. Enhanced features

### 5.1 Recommendation pipeline (`flaskr/tools/recommenders.py`)

1. **Time-decayed user-based kNN** — rating weights decay with age
   (half-life 365 days) before kNN-with-means is trained on the
   weighted matrix.
2. **TF-IDF + genre content model** — each movie's overview text is
   vectorised (1–2 grams, English stop-words, L2-normalised) and
   horizontally stacked with a weighted genre multi-hot block.
3. **Hybrid fusion + MMR re-ranking** — CF and CB scores are
   min-max normalised, combined linearly (`α = 0.6`), then diversified
   with MMR whose `λ` is bound to the navbar *Diversity* slider.
4. **Interest weighting** — *Interested* / *Not interested* marks from
   the onboarding flow add `+0.5` / `−0.3` to the item's final score.
5. **Dislike filtering** — items 👎'd by the user are suppressed in
   subsequent refreshes.

### 5.2 UI features (Enhanced template + `static/app.js`)

| Feature             | Where                         | What it does                                          |
|---------------------|-------------------------------|-------------------------------------------------------|
| Profile Switcher    | Navbar dropdown (P01–P06)     | Prefixes every cookie with the active profile id     |
| Haven't seen flow   | Onboarding cards              | Interested / Not interested / Skip instead of a star |
| Search              | Onboarding / main page        | Instant title search over the movie catalogue         |
| Why? explanations   | Badge on every rec card       | `/api/explain/<movieId>` returns top 3 similar likes |
| Diversity slider    | Navbar                        | Rebinds MMR `λ` (0 = relevance, 1 = diversity)       |
| Trending Now        | Above Recommended for You     | Horizontally scrollable strip of globally popular    |
| Your Taste          | Navbar modal                  | Chart.js radar + decade histogram + top TF-IDF words |
| Feedback Diary      | Navbar modal                  | Full Like / Dislike / Haven't-seen history           |

### 5.3 Relevant API endpoints

| Route                          | Purpose                                  |
|--------------------------------|------------------------------------------|
| `GET /`                        | Main page (served under current `UI_MODE`) |
| `GET /api/explain/<movieId>`   | JSON payload for the "Why?" popover      |
| `GET /api/search?q=...`        | Title search for the onboarding box      |
| `GET /api/profile/stats`       | Feeds the Your Taste dashboard           |
| `POST /api/interest`           | Records Interested / Not interested / Skip |

---

## 6. A/B test workflow

The study is a **within-subject A vs B** comparison with **6
participants** and **counter-balanced order** (AB / BA), producing
**12 sessions** total. Full protocol in `evaluation/procedure.md`.

### 6.1 One-time setup

```cmd
conda activate lab3
cd /d D:\path\to\Demo_materials
copy evaluation\results_template.csv evaluation\results.csv
```

Open `evaluation\results.csv` in an editor and keep only the header
row (delete any example rows).

Open **two** Anaconda Prompts with `lab3` active:
- **Window ①** runs the Flask server
- **Window ②** runs `collect_response.py` after each session

### 6.2 Session schedule (12 sessions, 2 server restarts)

| Phase | Server     | Sessions               | Reset between sessions?                |
|-------|------------|------------------------|----------------------------------------|
| A1    | Original   | P01, P03, P05 (A)      | Click **Clean All** before each       |
| B     | Enhanced   | P01, P03, P05 (B), then P02, P04, P06 (B) | Switch **Profile** dropdown; click **Reset** only when the same profile previously ran |
| A2    | Original   | P02, P04, P06 (A)      | Click **Clean All** before each       |

### 6.3 Per-session checklist (~10 min)

1. **Task 1 — Onboarding (3 min)**: pick ≥ 3 genres, rate or mark
   *Haven't seen* for ≥ 10 films until the progress bar reaches 10/10.
2. **Task 2 — Exploration (4 min)**: open 3 detail modals, 👍 two
   items, 👎 one item.
3. **Task 3 — Refinement (2 min)**: refresh recommendations; in the
   Enhanced variant try the Diversity slider and Your Taste dashboard.
4. **Rate Q1–Q8** on a 1–5 Likert scale (see `evaluation/questionnaire.md`).
5. Switch to **Window ②** and run:
   ```cmd
   python evaluation\collect_response.py
   ```
   Type `participant_id`, `order`, `condition`, the 8 Likert scores,
   and two short open-ended answers. The script appends a row to
   `evaluation/results.csv`.

> **Important**: score each session *immediately* after finishing —
> do not batch until the end, or you will lose discrimination.

---

## 7. Evaluation pipeline

`evaluation/` contains everything needed to analyse the collected data:

```
evaluation/
├── procedure.md             Full A/B protocol + task scripts
├── consent_form.md          Informed-consent template
├── questionnaire.md         Q1–Q8 Likert + Q9/Q10 open-ended
├── questionnaire_gform.txt  Ready-to-paste items for Google Forms
├── results_template.csv     Header-only CSV to copy into results.csv
├── results.csv              Your collected data (12 rows + header)
├── mock_results.csv         6 fake participants for a dry-run
├── collect_response.py      Interactive appender for results.csv
└── analyze_results.py       Paired t-test + Wilcoxon + Cohen's dz + plots
```

Once `results.csv` holds all 12 rows, generate the summary and figures:

```cmd
python evaluation\analyze_results.py evaluation\results.csv ^
    --out evaluation\results_summary.csv ^
    --figures-dir report\figures
```

Outputs:
- Terminal: Q1–Q8 table with `mean_A`, `mean_B`, `diff`, `t`, `p`, `dz`.
- `evaluation\results_summary.csv`: same content as CSV.
- `report\figures\fig_mean_bars.png`: A vs B mean bars per question.
- `report\figures\fig_paired_box.png`: paired box-plots per question.

For a dry-run without real data, substitute `mock_results.csv`.

An offline smoke test for the recommender (no browser needed):

```cmd
python scripts\smoke_test_recommenders.py
```

---

## 8. Dataset

Located at `flaskr/static/ml_data/` (MovieLens 1M-style CSVs).

`ratings.csv` columns:
- `userId` — user identifier
- `movieId` — movie identifier
- `rating` — 0.5–5.0 in half-star steps
- `timestamp` — seconds since epoch (newer = larger)

Convert timestamps with pandas:

```python
import pandas as pd
pd.to_datetime(1717665888, unit='s').strftime('%Y-%m-%d %H:%M:%S')
# -> '2024-06-06 09:24:48'
```

Other files: `movies.csv` (id / title / genres), `links.csv`
(id → TMDb), plus cached posters pulled by `flaskr/scrape.py`.

---

## 9. Troubleshooting

| Symptom                                       | Likely cause                                  | Fix                                                                 |
|-----------------------------------------------|-----------------------------------------------|---------------------------------------------------------------------|
| `ERR_CONNECTION_REFUSED` in browser           | Flask window was closed / Ctrl+C'd            | Restart via `scripts\run_enhanced.bat`                              |
| `Address already in use` on start             | Previous Flask process still holds port 5000  | `netstat -ano \| findstr :5000` → `taskkill /F /PID <pid>`          |
| Ctrl+C doesn't stop the server                | Watchdog reloader has a child process         | Press Ctrl+C twice, or Ctrl+Break, or close the terminal window    |
| `scikit-surprise` install fails via pip       | Needs C compiler                              | Use conda: `conda install -c conda-forge scikit-surprise`           |
| Blank page or Vue `[[...]]` shown literally   | Browser cached old template                   | Hard-refresh: **Ctrl+Shift+R**                                      |
| Original page shows *someone else's* ratings  | `profile_id` cookie leaked in from Enhanced  | Click **Clean All** — the Original template also wipes `profile_id` on load |
| `analyze_results.py` prints `t=NaN / p=NaN`   | All B−A diffs for one question are identical  | Adjust one or two participants' scores by ±1 so SD > 0              |
| Posters show *No Cover*                       | TMDb scrape cache miss for that movie         | Harmless; falls back to placeholder                                 |

---

## License

See `LICENSE` (MIT). Course assignment for HKBU COMP4135, 2026.
