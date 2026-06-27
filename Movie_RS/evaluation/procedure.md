# A/B Evaluation Procedure — Movie Recommender System

This document describes the experiment protocol used to compare the
**Original** demo (baseline) against the **Enhanced** version of the
Movie Recommender System developed for COMP4135.

## 1. Design

- **Design type**: within-subject A/B test with condition-order
  counter-balancing (Latin square).
- **Participants**: at least 6 per evaluation (assignment minimum).
  Recruit fellow classmates / friends who are ≥18 years old and use
  English comfortably.
- **Conditions**:
  1. **A — Original** (`REC_ALGO=original`, `UI_MODE=original`):
     user-based kNN recommendations, original demo UI.
  2. **B — Enhanced** (`REC_ALGO=enhanced`, `UI_MODE=enhanced`):
     time-decayed kNN + TF-IDF content-based hybrid with MMR
     diversification, redesigned UI with search, explanations,
     diversity slider, and feedback diary.
- **Counter-balancing**: half of the participants start with A then
  switch to B; the other half starts with B then A. Record the order
  in the result sheet.

## 2. Tasks for each participant

For each condition (A and B) the participant performs the same three
tasks. In the Enhanced UI, switch to the correct participant in the
navbar **Profile** drop-down (`P01`…`P06`) before the session starts;
this keeps each participant's ratings/likes isolated in its own cookie
namespace, so no manual Reset is required between participants.
Between the two conditions (A ↔ B) the same participant should still
press the **Reset** button so their second condition starts from a
blank slate.

1. **Onboarding**: select at least three favourite genres and rate
   at least 10 movies on a 1–5 scale.
2. **Exploration**: browse the generated recommendation list, inspect
   three different movies (open the detail view), and mark at least
   two as liked and one as not interested.
3. **Refinement**: regenerate the recommendations once after giving
   feedback and record whether the new list is better.

## 3. Metrics collected

After each condition the participant fills in the questionnaire in
`questionnaire.md` (paper, Google Form, or Qualtrics).  The same
questions are asked in both conditions; only the `condition` field
differs.

For each Likert item we collect a 1–5 score (1 = strongly disagree,
5 = strongly agree).  The full list:

| Code   | Dimension          | Item                                                                    |
|--------|--------------------|-------------------------------------------------------------------------|
| Q1     | Recommendation quality | "The recommended movies match my taste."                               |
| Q2     | Diversity             | "The list of recommendations is diverse enough."                         |
| Q3     | Novelty               | "I discovered movies I had not seen or considered before."               |
| Q4     | Transparency          | "I understood why each movie was recommended to me."                     |
| Q5     | Control               | "I felt in control of the recommendations (via feedback / sliders)."     |
| Q6     | Ease-of-use           | "The interface was easy to use."                                         |
| Q7     | Satisfaction          | "Overall, I am satisfied with the recommendations."                      |
| Q8     | Trust                 | "I trust the system's recommendations."                                  |

Two open-ended questions are also recorded:

- **Q9**: What did you like the most in this version?
- **Q10**: What would you change or improve?

## 4. Data logging

Every submission is appended to `evaluation/results.csv`. The CSV
header is already prepared in `results_template.csv`:

```
participant_id,order,condition,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,comment_like,comment_improve
```

- `participant_id`: P01 … P06+ (anonymous).
- `order`: `AB` or `BA` — which condition came first.
- `condition`: `A` (original) or `B` (enhanced).

## 5. Statistical analysis

Differences between the two conditions are analysed with a **paired
t-test** (or Wilcoxon signed-rank test when the normality assumption
fails) per question, because every participant provides matched A/B
scores. Run:

```
python evaluation/analyze_results.py evaluation/results.csv
```

The script prints means, standard deviations, the paired t-statistic,
p-value, and effect size (Cohen's dz) for every Likert item.

## 6. Running each condition

Before each session, from `Anaconda Prompt` activate the environment
and start the server in the requested mode:

```
conda activate lab3
cd d:\HKBU\year3-seme2\COMP4135\Demo_materials

# Condition A (original baseline)
set REC_ALGO=original
set UI_MODE=original
flask --app flaskr run

# Condition B (enhanced system)
set REC_ALGO=enhanced
set UI_MODE=enhanced
flask --app flaskr run
```

In PowerShell replace `set X=Y` with `$env:X="Y"`.

Ask the participant to open `http://127.0.0.1:5000` and click the
`Reset` button before the new condition starts, so cookies from the
previous condition are cleared.
