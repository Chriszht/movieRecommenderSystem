# 1. Cover Page

**Project title**: Enhancing a Movie Recommender System — Hybrid Scoring, Explainability, and User-Centred Feedback

**Course**: COMP4135 Recommendation and Search Techniques (2025–26, Semester 2)
**Institution**: Department of Computer Science, Hong Kong Baptist University
**Group name**: _<TODO: your group name>_
**Submission date**: _<TODO: YYYY-MM-DD>_

## Group Members

| # | Name | Student ID | Primary role |
|---|------|------------|--------------|
| 1 | _<TODO>_ | _<TODO>_ | _e.g., Algorithm (kNN + hybrid)_ |
| 2 | _<TODO>_ | _<TODO>_ | _e.g., UI / Vue front-end_ |
| 3 | _<TODO>_ | _<TODO>_ | _e.g., Evaluation & statistics_ |
| 4 | _<TODO>_ | _<TODO>_ | _e.g., Report & visualisation_ |

---

## Table of Contents

1. Cover Page (above)
2. System Description
3. User Interface Screenshots
4. Evaluation Procedure & Results
5. Individual Reflections
6. GenAI Usage Disclosure
7. References
8. Signed Group Participation Form (attached)
Appendices A–E

---

# 2. System Description (~1.5 pages)

## 2.1 Problem & baseline

The course provides a Flask + Vue + Bulma demo movie recommender built on a MovieLens subset (610 users, 9 742 movies). The baseline uses an item-based kNN over raw ratings and offers a minimalist UI (genre picker → 10-movie rating → "Recommended for You" / "Liked with Similar Items"). Five gaps motivated our enhancements:

1. **No time awareness** — all ratings are treated equally regardless of age.
2. **No content signal beyond genres** — movie overviews are ignored.
3. **No explanation** — users cannot see *why* a film was recommended.
4. **Coarse feedback** — only a binary Like button; no dislike, no "haven't seen", no detail view.
5. **No diversity control** — the top-N list often feels repetitive.

**Research questions** guiding the A/B evaluation:
- **RQ1 (algorithm)**: Does the enhanced hybrid pipeline produce recommendations that match user taste better than the baseline kNN?
- **RQ2 (interface)**: Does the redesigned UI improve perceived transparency, control, and diversity?

## 2.2 Architecture overview

```
Browser (Vue 3 + Chart.js + Bulma)
        │  HTTP + JSON
        ▼
Flask (flaskr/main.py)
    │
    ├── flaskr/tools/data_tool.py        — CSV loaders + poster cache
    └── flaskr/tools/recommenders.py
            ├── Time-decayed User-kNN        (A1)
            ├── TF-IDF + genre content model (A2)
            ├── Hybrid fusion + MMR          (A3)
            └── Explanation builder          (for "Why?" popover)
```

Two environment variables (`REC_ALGO`, `UI_MODE`) select between the **Original** baseline and the **Enhanced** version, enabling a clean within-subject A/B comparison (see `README.md` §3.2).

## 2.3 Data

| Source | Fields used | Where |
|---|---|---|
| MovieLens `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | CF model, decay weighting |
| MovieLens `movies.csv` | `movieId`, `title`, `genres` | Multi-hot genre block |
| TMDB (scraped) | `overview`, `cover_url`, `release_date` | TF-IDF text vectors, UI posters, decade histogram |

## 2.4 Baseline (inherited) features

| Feature | Description | File |
|---|---|---|
| Genre selection modal | Multi-select at first visit | `templates/index_original.html` |
| 10-movie cold-start rating | Stars on popular posters per chosen genre | `templates/index_original.html` |
| User-kNN recommendations | `KNNWithMeans` from `scikit-surprise` on raw ratings | `flaskr/main_original.py` |
| Item-based "similar to liked" | Cosine similarity over rating matrix | `flaskr/main_original.py` |
| Persistent Like list | Liked strip at the bottom | `templates/index_original.html` |
| Cookie-based state | All user state in browser cookies | same |

## 2.5 Enhanced features — algorithmic (`flaskr/tools/recommenders.py`)

1. **Time-decayed User-based kNN (A1).** Rating weights decay exponentially with a 365-day half-life:

   r'_{u,i} = r̄_u + (r_{u,i} − r̄_u) · exp(−ln 2 · Δt / H),   H = 365 d

   The decayed matrix feeds Pearson-similarity `KNNWithMeans`. *Rationale*: taste drifts; ignoring timestamps is a well-known baseline weakness (Koren, 2010).

2. **TF-IDF + genre content model (A2).** Each movie's overview is vectorised with `TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=2)`, L2-normalised, and horizontally stacked with a 2×-weighted genre multi-hot block. The user profile is the weighted mean of liked / highly-rated item vectors; candidates are scored by cosine similarity. *Rationale*: adds semantic signal so sparse or cold-start users still get useful results.

3. **Hybrid fusion + MMR diversity (A3).** CF and CB scores are min-max normalised and combined linearly (α = 0.6). The top candidates go through Maximal Marginal Relevance re-ranking:

   MMR(i) = (1 − λ) · s_i  −  λ · max_{j ∈ S} sim(i, j)

   where λ ∈ [0, 1] is bound to the Diversity slider. *Rationale*: balances relevance and novelty; gives the user explicit control (Carbonell & Goldstein, 1998).

4. **Interest weighting & dislike filtering.** *Interested* / *Not interested* onboarding marks add **+0.5 / −0.3** to the final score. Items flagged 👎 are suppressed in subsequent refreshes. *Rationale*: turns implicit onboarding signal into explicit scoring bias.

## 2.6 Enhanced features — UI / UX

Nine interactive components address specific gaps in the baseline:

| # | Feature | Requirement addressed | Endpoint / file |
|---|---|---|---|
| U1 | Catalogue **search box** in onboarding | B1 — richer preference elicitation | `GET /api/search` |
| U2 | **"Why?" explanations** (badge + popover) on every card | B2 — transparency | `GET /api/explain/<movieId>` |
| U3 | **👍 / 👎 feedback** + Feedback Diary drawer | B3 — richer feedback | cookie-based; `_modals.html` |
| U4 | **Detail modal** (poster + overview + genres + explanation) | B4 — richer information | `_modals.html` |
| U5 | **Diversity slider** in the navbar, re-ranks live | B5 — user control | `mmr_rerank` + cookie `diversity` |
| U6 | **"Haven't seen"** flow: *Interested / Not interested / Skip* | B1 + B3 — elicitation when user can't rate | `POST /api/interest` |
| U7 | **Trending Now** horizontally scrollable strip | B6 — popularity fallback | `main.py`, `index.html` |
| U8 | **Your Taste** dashboard (Chart.js radar + decade histogram + top-5 TF-IDF keywords) | B6 — transparency, reflection | `GET /api/profile/stats` |
| U9 | **Profile Switcher** (P01–P06 cookie prefixing) | Evaluation scaffold for within-subject A/B | `_pp()` in `main.py` |

## 2.7 Rationale summary

All four algorithmic techniques (A1–A3 + interest weighting) are taken directly from course-covered material (collaborative filtering, content-based filtering, hybridisation, diversity re-ranking). The interface enhancements are motivated by established HCI heuristics for recommender systems: explanations and transparency (Tintarev & Masthoff, 2007), user control (Pu & Chen, 2010), and the long-tail / novelty trade-off (Carbonell & Goldstein, 1998). U9 (Profile Switcher) is strictly an evaluation scaffold and does not alter recommendation logic; its presence is disclosed in §4.5 Limitations.

---

# 3. User Interface Screenshots

All PNGs should be saved under `report/figures/`. Capture at **1440 × 900** minimum (Chrome DevTools → Device mode → Responsive → set dimensions → Ctrl+Shift+P → *Capture full-size screenshot*). Each caption is a single sentence highlighting the **novelty vs the original design**.

## 3.1 Baseline (Original) — shown for comparison

Run `scripts\run_original.bat`, open `http://127.0.0.1:5000` in an incognito window, click **Clean All** first.

| # | Filename | What to capture | One-sentence caption |
|---|---|---|---|
| Fig 3.1 | `fig_orig_genres.png` | Genre selection modal on first visit | Baseline offers only a flat multi-select genre picker — no progress bar, no search. |
| Fig 3.2 | `fig_orig_rate.png` | 10-movie rating modal after picking genres | Baseline elicitation uses stars only, forcing users to rate titles they have never seen. |
| Fig 3.3 | `fig_orig_recs.png` | "Recommended for You" section after saving ratings | Baseline cards expose a single Like button with no explanation, detail view, or diversity control. |
| Fig 3.4 | `fig_orig_liked_similar.png` | "Liked with Similar Items" section + "Liked" strip at the bottom | Baseline similarity pane relies purely on item-level cosine over raw ratings. |

## 3.2 Enhanced (Proposed) — main deliverables

Run `scripts\run_enhanced.bat`, open `http://127.0.0.1:5000`, pick a profile, complete onboarding before capturing.

| # | Filename | What to capture | One-sentence caption |
|---|---|---|---|
| Fig 3.5 | `fig_nav.png` | Full top navigation bar, Profile dropdown visible | Adds Profile Switcher, Reset, Genres, Your Taste, Feedback Diary, and a live Diversity slider — none exist in the baseline. |
| Fig 3.6 | `fig_onboarding_search.png` | Onboarding with the search box typed "inception" and results shown | Introduces catalogue-wide search so users can rate films they actually know, not just the 10 default posters. |
| Fig 3.7 | `fig_havent_seen.png` | "Haven't seen" modal open showing *Interested / Not interested / Skip* | Lets cold-start users express preference without rating, removing the baseline's forced-star requirement. |
| Fig 3.8 | `fig_progress.png` | Onboarding progress bar at "10 / 10" | Visualises the cold-start threshold in real time; the baseline gives no progress feedback at all. |
| Fig 3.9 | `fig_trending.png` | Main page showing Trending Now horizontal strip above Recommended for You | Adds a globally-popular fallback strip that the baseline lacks entirely. |
| Fig 3.10 | `fig_recs_why.png` | Recommended for You with yellow "Why?" badges on each card | Every card now carries an inline explanation badge — baseline cards carry none. |
| Fig 3.11 | `fig_why_popover.png` | "Why?" popover expanded over one card | Opens a modal listing the top-3 similar liked items that triggered the recommendation. |
| Fig 3.12 | `fig_detail_modal.png` | Detail modal open (poster + overview + genres + explanation + Like/Dislike) | Replaces the baseline's flat card with a full detail view that supports informed feedback. |
| Fig 3.13 | `fig_diversity_side.png` | Two side-by-side recommendation lists at **λ = 0.1** vs **λ = 0.9** | Demonstrates live MMR re-ranking driven by the Diversity slider — a control the baseline does not provide. |
| Fig 3.14 | `fig_taste_radar.png` | Your Taste modal — Chart.js radar of the user's 20 genre averages | Offers a reflection dashboard; the baseline exposes no user-level analytics. |
| Fig 3.15 | `fig_taste_decade.png` | Your Taste — decade histogram + Top-5 TF-IDF keywords | Extends the dashboard with temporal and semantic taste summaries. |
| Fig 3.16 | `fig_diary.png` | Feedback Diary drawer listing Likes / Dislikes / Haven't-seen items | Provides an auditable history of user feedback; the baseline has no such view. |
| Fig 3.17 | `fig_profile_dropdown.png` | Profile dropdown open showing P01–P06 | Adds per-profile cookie isolation so multi-participant studies need no Reset between users. |

## 3.3 Capture checklist (avoid rework)

1. Hard-refresh (Ctrl+Shift+R) before each capture so you get the latest CSS/JS.
2. Hide the dev tools panel and any personal bookmarks bar before the snap.
3. Keep window size consistent across all figures for a uniform report look.
4. Save as PNG (no JPEG artefacts on text), filename exactly as listed above.
5. If a figure contains private data (e.g., a real name), blur it in Snipping Tool → Paint → pixelate.

---

# 4. Evaluation Procedure & Results (~2.5 pages)

## 4.1 Participants & evidence of participation

We recruited **N = 6 participants (P01–P06)** from _<TODO: e.g., HKBU Year-3 classmates>_. All signed the consent form before their session (template in `evaluation/consent_form.md`; scans in **Appendix A**). On-site evaluation photos are optional evidence and are stored under `report/figures/photos/`.

**Table 4-1. Participant demographics.**

| Anon. ID | Gender | Age | Streaming-service use | Prior recsys awareness | Consent |
|---|---|---|---|---|---|
| P01 | _<M/F/Other>_ | _<18-24>_ | _<≥1/week>_ | _<Yes/No>_ | ✓ |
| P02 | ... | ... | ... | ... | ✓ |
| P03 | ... | ... | ... | ... | ✓ |
| P04 | ... | ... | ... | ... | ✓ |
| P05 | ... | ... | ... | ... | ✓ |
| P06 | ... | ... | ... | ... | ✓ |

Aggregate: _<TODO: e.g., 4 F / 2 M; median age 21; all weekly users of at least one streaming service>_.

## 4.2 Procedure & measurements

**Design**: within-subject A vs B, counter-balanced order (3 × AB, 3 × BA), 12 total sessions (6 participants × 2 conditions). Full protocol in `evaluation/procedure.md`.

**Per-session tasks (~10 min each)** — identical across conditions:
- **T1 Onboarding**: select ≥ 3 genres; provide ratings or "Haven't seen" marks for ≥ 10 films (progress bar must reach 10/10).
- **T2 Exploration**: open 3 detail views, give 2 Likes and 1 Dislike.
- **T3 Refinement**: refresh the recommendation list; in condition B also move the Diversity slider and open the Your Taste dashboard.

**Measurements** — 8 Likert items (1 = strongly disagree, 5 = strongly agree) plus 2 open-ended questions (`evaluation/questionnaire.md`):

| # | Dimension             | Example phrasing                                 |
|---|-----------------------|--------------------------------------------------|
| Q1 | Recommendation quality | "The suggestions matched my taste."             |
| Q2 | Diversity              | "The list offered varied options."              |
| Q3 | Novelty                | "I discovered films I didn't already know."     |
| Q4 | Explainability         | "I understood why each item was recommended."   |
| Q5 | Control                | "I could influence the results."                |
| Q6 | Usability              | "The interface was easy to use."                |
| Q7 | Overall satisfaction   | "I would use this system again."                |
| Q8 | Trust                  | "I would follow its suggestions."               |
| Q9 | Qualitative (+)        | "What did you like most?"                       |
| Q10 | Qualitative (−)       | "What would you change?"                        |

**Statistical tests** (all computed by `evaluation/analyze_results.py`):
- Paired **t-test** (primary) on each Likert item's B−A difference.
- **Wilcoxon signed-rank** as a non-parametric robustness check.
- **Cohen's d_z** for effect size (0.2 small, 0.5 medium, 0.8 large).

## 4.3 Quantitative results

Regenerate with:

```
python evaluation/analyze_results.py evaluation/results.csv \
    --out evaluation/results_summary.csv \
    --figures-dir report/figures
```

**Table 4-2. Paired comparison (N = 6) — Enhanced (B) vs Original (A).** Copy numbers verbatim from `evaluation/results_summary.csv`.

| Item | Mean A | SD A | Mean B | SD B | Δ (B−A) | t(5) | p (two-tailed) | Wilcoxon p | Cohen's d_z |
|---|---|---|---|---|---|---|---|---|---|
| Q1 Quality        | _<f>_ | _<f>_ | _<f>_ | _<f>_ | _<f>_ | _<f>_ | _<f>_ | _<f>_ | _<f>_ |
| Q2 Diversity      | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Q3 Novelty        | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Q4 Explainability | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Q5 Control        | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Q6 Usability      | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Q7 Satisfaction   | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Q8 Trust          | ... | ... | ... | ... | ... | ... | ... | ... | ... |

- **Figure 4-1** (`report/figures/fig_mean_bars.png`) — mean scores A vs B per Likert item.
- **Figure 4-2** (`report/figures/fig_paired_box.png`) — paired box-plots per Likert item.

**Narrative template** (replace bracketed values with your actual findings): "Condition B scored significantly higher than A on Q4 Explainability (M_B = _<f>_ vs M_A = _<f>_, t(5) = _<f>_, p = _<f>_, d_z = _<f>_) and Q5 Control (…). No significant difference was observed on Q6 Usability (p = _<f>_), consistent with the extra complexity introduced by the enhanced UI."

## 4.4 Qualitative findings

Summarise Q9 / Q10 into **3–5 themes per condition** and quote representative responses with their participant ID and condition label:

- **Liked in B**: _"The 'Why?' explanations made the recommendations feel trustworthy." (P01, B)_
- **Liked in A**: _"Simple interface, easy to rate movies quickly." (P03, A)_
- **Criticism of B**: _"Too many features on one page; could feel overwhelming at first." (P05, B)_
- **Criticism of A**: _"Recommendations felt repetitive and unexplained." (P02, A)_

## 4.5 Discussion & limitations

- **RQ1 (algorithm)**: _<Yes / partially / no — tie to Q1, Q3 statistics>_.
- **RQ2 (interface)**: _<Tie to Q2, Q4, Q5 statistics>_.
- **Limitations**:
  1. N = 6 limits statistical power; results should be treated as exploratory.
  2. Possible novelty bias favouring condition B.
  3. Single-session measurement cannot assess long-term satisfaction.
  4. Counter-balanced AB/BA ordering mitigates but does not eliminate learning effects.
  5. All participants were university students from a single institution — limited demographic generalisability.
  6. The Profile Switcher is present only in condition B's UI; this was disclosed to participants and is a known asymmetry, but cannot fully be ruled out as a confound.

---

# 5. Individual Reflections

> **Assessed individually.** Each group member must fill in their own block below. Use the exact template (role paragraph + two observations/suggestions paragraphs) so that the grader can locate each member's contribution unambiguously.

## 5.1 Member 1 — _<Name>_, _<Student ID>_

**(a) Role** _(~one paragraph, ~120 words)_.
_<TODO: Describe what you personally did. E.g.: "I was responsible for the TF-IDF content model and the 'Why?' explanation pipeline. I implemented `build_content_matrix` and `content_based_scores` in `flaskr/tools/recommenders.py`, tuned the stop-word list and `min_df`, and wrote the `/api/explain/<movieId>` endpoint that powers the popover. I served as the experimenter for participants P01–P03 and authored §2.5 and §4.3 of the report.">_

**(b) Observations & suggestions** _(~two paragraphs)_.

_Paragraph 1 — issues observed_: _<TODO: Pick 2–3 concrete issues you noticed. Examples: dense navbar on small screens; TF-IDF profile is noisy when the user has fewer than 5 likes; the Why? popover sometimes cites the same anchor item across many recommendations; single-session evaluation cannot reveal repeated-exposure fatigue.>_

_Paragraph 2 — proposed enhancements_: _<TODO: For each issue above, suggest a concrete fix. Examples: replace TF-IDF with sentence-transformer embeddings for richer semantics; add a guided onboarding tour and collapse advanced controls behind an "Expert mode" toggle; log implicit dwell-time and scroll signals to de-bias the offline evaluation; introduce a lightweight sequential model such as SASRec to capture within-session dynamics.>_

## 5.2 Member 2 — _<Name>_, _<Student ID>_

**(a) Role**: _<TODO: same template as 5.1>_.

**(b) Observations & suggestions**: _<TODO: two paragraphs, same template>_.

## 5.3 Member 3 — _<Name>_, _<Student ID>_

**(a) Role**: _<TODO>_.

**(b) Observations & suggestions**: _<TODO>_.

## 5.4 Member 4 — _<Name>_, _<Student ID>_

**(a) Role**: _<TODO>_.

**(b) Observations & suggestions**: _<TODO>_.

_(Duplicate or delete sub-sections to match the actual group size.)_

---

# 6. GenAI Usage Disclosure

| Tool | Version / model | Purpose | Extent of use | Verification done by humans |
|---|---|---|---|---|
| _e.g., ChatGPT_ | _GPT-4o (2025-03)_ | Brainstorming questionnaire wording; drafting report prose | ~15 prompts; all outputs rewritten | Co-authors reviewed phrasing against Likert best practices; cross-checked against the course slides |
| _e.g., GitHub Copilot_ | _VS Code extension_ | Line-level autocomplete in `recommenders.py` and `app.js` | Suggestions accepted only when trivially correct | Every accepted line was run through `scripts/smoke_test_recommenders.py` |
| _e.g., Augment Agent_ | _Claude Opus 4.7 base_ | Implementation of enhanced features (Haven't-seen, Trending Now, Your Taste, Profile Switcher) and scaffolding of the A/B evaluation pipeline | Paired-programming sessions over ~2 weeks | Each diff manually reviewed; end-to-end A/B run performed before submission |

**Policy statement**: no GenAI-produced text, code, or figure was submitted without being read, edited, and verified by a human group member. All statistical analysis code (`evaluation/analyze_results.py`) was manually cross-checked against the `scipy.stats` documentation. No GenAI tool was involved in the participant recruitment, consent acquisition, or the running of the user-study sessions themselves.

---

# 7. References

## 7.1 Academic references

1. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. *IEEE Transactions on Knowledge and Data Engineering*, 17(6), 734–749.
2. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based re-ranking for reordering documents and producing summaries. *SIGIR '98*, 335–336.
3. Ding, Y., & Li, X. (2005). Time weight collaborative filtering. *CIKM '05*, 485–492.
4. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM TiiS*, 5(4), 1–19.
5. Herlocker, J. L., Konstan, J. A., & Riedl, J. (2000). Explaining collaborative filtering recommendations. *CSCW '00*, 241–250.
6. Koren, Y. (2010). Collaborative filtering with temporal dynamics. *Communications of the ACM*, 53(4), 89–97.
7. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8), 30–37.
8. Pu, P., Chen, L., & Hu, R. (2011). A user-centric evaluation framework for recommender systems. *RecSys '11*, 157–164.
9. Ricci, F., Rokach, L., & Shapira, B. (Eds.) (2015). *Recommender Systems Handbook* (2nd ed.). Springer.
10. Tintarev, N., & Masthoff, J. (2007). A survey of explanations in recommender systems. *ICDEW '07*, 801–810.

## 7.2 Software, libraries, and datasets

- **Flask 3.0** & **Werkzeug 3.0** — web framework (<https://flask.palletsprojects.com>)
- **scikit-learn 1.4** — `TfidfVectorizer`, `cosine_similarity`, `MinMaxScaler`
- **scikit-surprise 1.1** — `KNNWithMeans`
- **pandas 2.1**, **numpy 1.26**, **scipy 1.11** — data + statistics
- **Vue 3**, **Bulma 0.9.4**, **Chart.js**, **Font Awesome** — front-end UI
- **MovieLens** dataset (GroupLens, University of Minnesota)
- **The Movie Database (TMDB)** API — posters, overviews, release dates

---

# 8. Signed Group Participation Form

See **Appendix D** for the scanned copy of `Group_Participation.doc` signed by all members.

> **Note from the assignment brief**: "Assignments will NOT be graded without a participation form signed by all group members."

---

# Appendices

## Appendix A — Signed consent forms

Scanned PDFs of the per-participant consent forms (template: `evaluation/consent_form.md`). File naming: `appendix_a_consent_P01.pdf` … `appendix_a_consent_P06.pdf`.

## Appendix B — Questionnaire

Full text of the Q1–Q10 instrument is in `evaluation/questionnaire.md`. A Google-Forms-ready version is in `evaluation/questionnaire_gform.txt`.

## Appendix C — Raw results

Attach `evaluation/results.csv` (13 rows = 1 header + 12 sessions). The processed summary `evaluation/results_summary.csv` is the source for Table 4-2.

## Appendix D — Signed Group Participation Form

Attach `Group_Participation_signed.pdf`.

## Appendix E — On-site evaluation photos (optional)

Place photos in `report/figures/photos/` with faces blurred. Useful as additional evidence of on-site participation.

## Appendix F — Work-distribution statement

_(Short paragraph or table listing each member's contribution areas: algorithm, UI, evaluation, report, etc. Must be consistent with §5 individual reflections.)_

