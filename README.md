# RASE — Regime-Aware Stacked Ensemble for AQI Forecasting in 63 Indian Cities

Code and results for the paper:

> **Regime-Aware Explainable Stacked Ensemble for Multi-Horizon Air Quality Index Forecasting Across 63 Indian Cities**
> Adbhut Prakash Dwivedi and Raghuraj Singh
> Department of Computer Science and Engineering, Harcourt Butler Technical University, Kanpur

This repository reproduces every table and figure in the paper from the public CPCB daily
AQI bulletin record. No proprietary data is required.

---

## What the method does

Four tree ensembles predict the **change** in AQI, Δ = AQI(t+h) − AQI(t), and are combined
by a linear meta-learner with non-negative coefficients. A second, identical stack is
trained with each Very Poor or Severe target day weighted five times (λ = 4). A gate reads
the index on the issue day and routes to the episode stack when AQI(t) ≥ 150, otherwise to
the standard stack. The gate depends only on information available when the forecast is
issued, so the composition is a valid forecasting rule.

Because both meta-learners are linear with non-negative coefficients and the gate selects
exactly one stream per city-day, the Shapley values of the complete gated ensemble are the
coefficient-weighted sums of the learners' exact tree Shapley values — the decomposition is
exact rather than sampled, and additivity is verified numerically in the notebook.

---

## Repository layout

```
RASE_v2_revised.ipynb      Standalone notebook: runs end to end, Colab-ready
src/
  core.py                  Bulletin parsing with a full exclusion audit, the 32 features,
                           the four learners, metrics, and the HAC Diebold-Mariano test
  stage_a.py               Main run: both streams, all 17 models, three horizons
  stage_b.py               Tables: accuracy, significance, categories, warning skill,
                           per-city results, component ablation, protocol comparison
  stage_c.py               Design grid, feature ablation, rolling origin, out-of-fold
                           stacking protocol, conformal intervals, unseen-city experiment
  stage_d.py               Exact Shapley decomposition of the gated ensemble
  figures.py               All figures at 600 dpi
results/
  predictions/             Model forecasts for every test city-day, h = 1, 2, 3
  tables/                  Every table in the paper, as CSV
  figures/                 Figures 1-5 at 600 dpi
```

The notebook is generated from `src/`, so the two cannot drift apart.

---

## Data

Bulletins come from the open repository
[athulvis/AQI_scraper](https://github.com/athulvis/AQI_scraper), which converts the official
CPCB PDF bulletins into tables.

**The analysis is pinned to commit `82b3914`.** That repository is updated as new bulletins
are published, so without pinning the record changes under you. The notebook checks the
commit out automatically.

Cleaning rules, in the order applied, with the count each removes printed by the code:

| Rule | Effect |
|---|---|
| Unparsable rows | AQI, category or city field unreadable |
| Category-range check | Reported AQI outside the numeric range of its own reported category |
| Missing city name | City field empty after parsing |
| Name normalisation | Spellings differing only by case or whitespace merged to the most frequent |
| De-duplication | Same city and date reported twice; later record kept |
| City selection | 63 cities with at least 2,500 daily records retained |

No AQI value is interpolated, smoothed or imputed at any stage. A missing city-day stays
missing, and rolling features carry minimum-observation requirements so they are never
computed from near-empty windows.

---

## Running it

### Colab or Jupyter

Open `RASE_v2_revised.ipynb` and run all cells. The first cell clones and pins the data and
installs `shap`, `xgboost`, `lightgbm` and `catboost` if they are absent.

### Locally

```bash
pip install -r requirements.txt
jupyter notebook RASE_v2_revised.ipynb
```

Set `HORIZONS = [1]` and turn off the heavier flags in the configuration cell for a quick
pass.

**Run time** on two CPU cores: about 30 minutes for the main run over three horizons, plus
roughly 40 minutes for the full set of revision experiments. The exact-Shapley stage is the
slowest part and is controlled by `RUN_SHAP`.

### Verifying the tables without retraining

Every table in the paper is a function of `results/predictions/`, which is included here.
To check any number without fitting a model, load the predictions and call the functions in
`src/stage_b.py`.

---

## Reproducibility

Random seeds are fixed (`SEED = 42`) and the data snapshot is pinned, so a repeat run on the
same machine reproduces the published forecasts to within 1e-13 AQI points. Across different
hardware and thread counts, differences in floating-point reduction order can move reported
RMSE values in the final decimal place; this is expected for histogram-based boosting and
does not change any conclusion in the paper.

---

## Scope and limitations

The model uses CPCB bulletin values alone. No meteorological, emission or satellite
covariate enters the feature set, so the results describe what is achievable from the public
bulletin record and are not an upper bound on forecast skill.

Two results are reported in the paper against the method's own interest and are reproducible
here: the regime gate is the smallest of the three components, and the extreme-day advantage
does not transfer to cities absent from training.

---

## Citation

A BibTeX entry will be added once the proceedings reference is
assigned.

## License

Code is released under the MIT License (see `LICENSE`). The bulletin data are the property
of the Central Pollution Control Board and are redistributed here only as model outputs, not
as raw bulletins.
