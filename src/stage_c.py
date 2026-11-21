"""Stage C - the experiments the reviewers asked for that need model fitting.

C1  design-selection grid on the 2024 validation year (full lambda x tau grid)
C2  feature-group ablation on the test period (proxy learner)
C3  rolling-origin evaluation (proxy learner)
C4  lambda/tau sensitivity measured on the TEST period, by year and by region
C5  Protocol B: chronological out-of-fold stacking, design year separated from
    the calibration year  (reviewer 2, point 3)
C6  conformal prediction: marginal and Mondrian (regime-conditional) intervals
    with category-wise coverage and width  (reviewer 2, point 7)
C7  unseen-city generalisation: grouped 5-fold over cities  (reviewer 2, point 8)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core
from core import FEATURES, GROUPS, LABEL, make, SEED, metrics, aqi_category, CAT_NAMES
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

OUT = os.environ.get("RASE_OUT", "results")
RES = OUT
os.makedirs(RES, exist_ok=True)
LAM, TAU_ = 4, 150
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, np.clip(b, 0, 500))))


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ C1 + C4
def c1_c4(D1):
    X = D1[FEATURES].astype(float)
    y = D1.target.values.astype(float)
    l0 = D1.lag0.values.astype(float)
    d = y - l0
    tr, va, te = [(D1.split == s).values for s in ("train", "val", "test")]
    trva = tr | va
    LAMS = [0, 1, 2, 4, 6, 8]
    TAUS = [0, 100, 150, 200, 250, 300]

    # --- C1: fitted on train, evaluated on the 2024 validation year
    yv, av = y[va], l0[va]
    dS_v = make("hgb").fit(X[tr], d[tr]).predict(X[va])
    lvl_v = make("hgb").fit(X[tr], y[tr]).predict(X[va])
    dE_v = {}
    for lam in LAMS:
        if lam == 0:
            dE_v[lam] = dS_v
        else:
            dE_v[lam] = make("hgb").fit(X[tr], d[tr], sample_weight=1.0 + lam * (y[tr] > 300)).predict(X[va])
        log(f"   C1 lambda={lam} done")
    rows = {"Persistence": av, "AQI-level target": lvl_v, "AQI-change target": av + dS_v}
    for lam in LAMS:
        if lam:
            rows[f"Change target, episode weight lambda = {lam}"] = av + dE_v[lam]
    for tau in [100, 150, 200, 250]:
        rows[f"Regime gate tau = {tau}, lambda = 4"] = av + np.where(av >= tau, dE_v[4], dS_v)
    r = lambda p, m: rmse(yv[m], p[m])
    T1 = pd.DataFrame({k: [r(p, yv >= 0), r(p, yv > 200), r(p, yv > 300)] for k, p in rows.items()},
                      index=["RMSE (all days)", "RMSE (AQI > 200)", "RMSE (AQI > 300)"]).T.round(2)
    T1.to_csv(f"{RES}/c1_table1_validation.csv")

    # full grid on validation
    grid_v = []
    for lam in LAMS:
        for tau in TAUS:
            p = av + np.where(av >= tau, dE_v[lam], dS_v)
            grid_v.append({"lambda": lam, "tau": tau, "RMSE all": r(p, yv >= 0),
                           "RMSE >200": r(p, yv > 200), "RMSE >300": r(p, yv > 300)})
    pd.DataFrame(grid_v).to_csv(f"{RES}/c4_grid_validation.csv", index=False)

    # --- C4: the same grid, now measured on the untouched test period
    yt, at = y[te], l0[te]
    dS_t = make("hgb").fit(X[trva], d[trva]).predict(X[te])
    dE_t = {}
    for lam in LAMS:
        if lam == 0:
            dE_t[lam] = dS_t
        else:
            dE_t[lam] = make("hgb").fit(X[trva], d[trva], sample_weight=1.0 + lam * (y[trva] > 300)).predict(X[te])
        log(f"   C4 lambda={lam} done")
    sub = D1.loc[te].reset_index(drop=True)
    yr = sub.TargetDate.dt.year.values
    igp = sub.City.isin(core.IGP).values
    rows_t = []
    for lam in LAMS:
        for tau in TAUS:
            p = at + np.where(at >= tau, dE_t[lam], dS_t)
            rows_t.append({"lambda": lam, "tau": tau,
                           "RMSE all": rmse(yt, p), "RMSE >200": rmse(yt[yt > 200], p[yt > 200]),
                           "RMSE >300": rmse(yt[yt > 300], p[yt > 300]),
                           "RMSE >300 2025": rmse(yt[(yt > 300) & (yr == 2025)], p[(yt > 300) & (yr == 2025)]),
                           "RMSE >300 2026": rmse(yt[(yt > 300) & (yr == 2026)], p[(yt > 300) & (yr == 2026)]),
                           "RMSE >300 IGP": rmse(yt[(yt > 300) & igp], p[(yt > 300) & igp]),
                           "RMSE >300 non-IGP": rmse(yt[(yt > 300) & ~igp], p[(yt > 300) & ~igp])
                           if ((yt > 300) & ~igp).sum() > 5 else np.nan})
    pd.DataFrame(rows_t).round(2).to_csv(f"{RES}/c4_grid_test.csv", index=False)
    log("C1/C4 done")


# ------------------------------------------------------------------ C2 + C3
def c2_c3(D1):
    X = D1[FEATURES].astype(float)
    y = D1.target.values
    l0 = D1.lag0.values
    d = y - l0
    trva, te = (D1.split != "test").values, (D1.split == "test").values

    def proxy(fs):
        return metrics(y[te], l0[te] + make("hgb").fit(X.loc[trva, fs], d[trva]).predict(X.loc[te, fs]))

    sets = {"All features": FEATURES,
            **{f"Without {g.lower()}": [f for f in FEATURES if f not in v] for g, v in GROUPS.items()},
            "Recent history only": GROUPS["Recent history"]}
    T4 = pd.DataFrame({k: proxy(v) for k, v in sets.items()}).T[["RMSE", "MAE", "RMSE AQI>200", "RMSE AQI>300"]]
    T4["Change in RMSE (%)"] = 100 * (T4.RMSE / T4.loc["All features", "RMSE"] - 1)
    T4.round(2).to_csv(f"{RES}/c2_ablation.csv")
    log("C2 done")

    out = {}
    for name, (s, e) in {"2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"),
                         "2024": ("2024-01-01", "2024-12-31"),
                         "2025-26": ("2025-01-01", core.TEST_END)}.items():
        D = D1.drop(columns="city_mean_aqi")
        trn = D.TargetDate < s
        tst = (D.TargetDate >= s) & (D.TargetDate <= e)
        D = D.merge(D[trn].groupby("City").lag0.mean().rename("city_mean_aqi"),
                    left_on="City", right_index=True, how="left")
        Xr = D[FEATURES].astype(float)
        dd = (D.target - D.lag0).values
        p = np.clip(D.lag0.values[tst.values] + make("hgb").fit(Xr[trn.values], dd[trn.values]).predict(Xr[tst.values]), 0, 500)
        o = D.target.values[tst.values]
        out[name] = {"n test days": int(tst.sum()),
                     "RMSE gain vs persistence (%)": 100 * (1 - np.sqrt(mean_squared_error(o, p)) /
                                                            np.sqrt(mean_squared_error(o, D.lag0.values[tst.values])))}
    pd.DataFrame(out).T.round(1).to_csv(f"{RES}/c3_rolling_origin.csv")
    log("C3 done")


# ------------------------------------------------------------------ C5
def c5_protocol_b(D1, h=1):
    """Chronological out-of-fold stacking.  Base learners are fitted on all
    earlier years only and predict the next year; the meta-learner is fitted on
    those out-of-fold predictions.  lambda and tau are selected on 2023 alone,
    leaving 2024 free for conformal calibration."""
    X = D1[FEATURES].astype(float)
    y = D1.target.values.astype(float)
    l0 = D1.lag0.values.astype(float)
    d = y - l0
    yr = D1.TargetDate.dt.year.values
    te = (D1.split == "test").values
    names = ("rf", "et", "gbr", "hgb")
    # the first fold needs two full years of history so that the one-year lag
    # exists in the fitting set; the 2019 fold is therefore not used
    oof_years = [2020, 2021, 2022, 2023, 2024]
    oof_mask = np.isin(yr, oof_years) & ~te
    OOF = {w: {n: np.full(len(D1), np.nan) for n in names} for w in ("std", "epi")}
    for yy in oof_years:
        fit = (yr < yy) & ~te
        pred = (yr == yy) & ~te
        if fit.sum() < 5000 or pred.sum() == 0:
            continue
        for n in names:
            t = time.time()
            OOF["std"][n][pred] = make(n).fit(X[fit], d[fit]).predict(X[pred])
            w = 1.0 + LAM * (y[fit] > 300)
            OOF["epi"][n][pred] = make(n).fit(X[fit], d[fit], sample_weight=w).predict(X[pred])
            log(f"   C5 OOF {yy} {n} {time.time()-t:.0f}s")
    res = {}
    for w in ("std", "epi"):
        ok = oof_mask & ~np.isnan(OOF[w]["rf"])
        M = np.column_stack([OOF[w][n][ok] for n in names])
        sw = None if w == "std" else 1.0 + LAM * (y[ok] > 300)
        meta = LinearRegression(positive=True).fit(M, d[ok], sample_weight=sw)
        res[w] = {"coef": meta.coef_.tolist(), "intercept": float(meta.intercept_),
                  "n_oof": int(ok.sum())}
    # tau selected on 2023 out-of-fold predictions only
    sel = (yr == 2023) & ~te & ~np.isnan(OOF["std"]["rf"])
    mS = np.column_stack([OOF["std"][n][sel] for n in names])
    mE = np.column_stack([OOF["epi"][n][sel] for n in names])
    pS = np.asarray(res["std"]["coef"]) @ mS.T + res["std"]["intercept"]
    pE = np.asarray(res["epi"]["coef"]) @ mE.T + res["epi"]["intercept"]
    best = None
    for tau in [0, 100, 150, 200, 250, 300]:
        p = l0[sel] + np.where(l0[sel] >= tau, pE, pS)
        sev = y[sel] > 300
        sc = rmse(y[sel][sev], p[sev]) if sev.sum() > 20 else np.inf
        cap = rmse(y[sel], p) <= 1.02 * rmse(y[sel], l0[sel] + pS)
        if cap and (best is None or sc < best[1]):
            best = (tau, sc)
    tau_b = best[0] if best else TAU_
    res["tau_selected_on_2023"] = int(tau_b)
    json.dump(res, open(f"{RES}/c5_protocol_b_meta.json", "w"), indent=1)
    np.save(f"{RES}/c5_oof_std.npy", np.column_stack([OOF['std'][n] for n in names]))
    np.save(f"{RES}/c5_oof_epi.npy", np.column_stack([OOF['epi'][n] for n in names]))
    log("C5 done, tau selected on 2023 =", tau_b)


# ------------------------------------------------------------------ C6
def c6_conformal(D1):
    """Marginal and Mondrian conformalized quantile regression.
    The Mondrian version conditions on the CPCB category of the issue day,
    which is known when the forecast is made, so validity is preserved."""
    X = D1[FEATURES].astype(float)
    y = D1.target.values.astype(float)
    l0 = D1.lag0.values.astype(float)
    d = y - l0
    tr, va, te = [(D1.split == s).values for s in ("train", "val", "test")]
    q = {}
    for al in (0.1, 0.9):
        t = time.time()
        q[al] = HistGradientBoostingRegressor(loss="quantile", quantile=al, max_iter=600,
                                              learning_rate=0.05, max_leaf_nodes=48,
                                              min_samples_leaf=40, random_state=SEED).fit(X[tr], d[tr])
        log(f"   C6 quantile {al} {time.time()-t:.0f}s")
    lo = lambda m: l0[m] + q[0.1].predict(X[m])
    hi = lambda m: l0[m] + q[0.9].predict(X[m])
    lov, hiv, yv = lo(va), hi(va), y[va]
    lot, hit, yt = lo(te), hi(te), y[te]
    sc_v = np.maximum(lov - yv, yv - hiv)

    def conf_q(s):
        n = len(s)
        return np.quantile(s, min(1.0, np.ceil((n + 1) * 0.8) / n)) if n > 10 else np.nan

    Qm = conf_q(sc_v)
    # Mondrian bins on the issue-day category (lag0)
    bin_v, bin_t = aqi_category(l0[va]), aqi_category(l0[te])
    Qb = {}
    for b in range(6):
        m = bin_v == b
        Qb[b] = conf_q(sc_v[m]) if m.sum() > 50 else Qm
    Qt_m = np.full(len(yt), Qm)
    Qt_b = np.array([Qb.get(b, Qm) for b in bin_t])

    # third variant: condition on the predicted category.  The point forecast is
    # a function of information available when the forecast is issued, so this
    # conditioning is admissible and targets the regime the forecast implies
    # rather than the one the issue day was in.
    cen_v = 0.5 * (lov + hiv)
    cen_t = 0.5 * (lot + hit)
    pb_v, pb_t = aqi_category(cen_v), aqi_category(cen_t)
    Qp = {}
    for b in range(6):
        m = pb_v == b
        Qp[b] = conf_q(sc_v[m]) if m.sum() > 50 else Qm
    Qt_p = np.array([Qp.get(b, Qm) for b in pb_t])

    rows = []
    cy = aqi_category(yt)
    for nm, Qt in (("Marginal", Qt_m),
                   ("Mondrian (by issue-day category)", Qt_b),
                   ("Mondrian (by predicted category)", Qt_p)):
        L, U = lot - Qt, hit + Qt
        cov = (yt >= L) & (yt <= U)
        rows.append({"Method": nm, "Subset": "All test days", "n": len(yt),
                     "Coverage (%)": 100 * cov.mean(), "Mean width": float((U - L).mean())})
        for b, nmc in enumerate(CAT_NAMES):
            m = cy == b
            if m.sum() >= 20:
                rows.append({"Method": nm, "Subset": nmc, "n": int(m.sum()),
                             "Coverage (%)": 100 * cov[m].mean(), "Mean width": float((U - L)[m].mean())})
        for thr in (200, 300):
            m = yt > thr
            rows.append({"Method": nm, "Subset": f"AQI > {thr}", "n": int(m.sum()),
                         "Coverage (%)": 100 * cov[m].mean(), "Mean width": float((U - L)[m].mean())})
    unc = (yt >= lot) & (yt <= hit)
    rows.append({"Method": "Uncalibrated quantiles", "Subset": "All test days", "n": len(yt),
                 "Coverage (%)": 100 * unc.mean(), "Mean width": float((hit - lot).mean())})
    pd.DataFrame(rows).round(2).to_csv(f"{RES}/c6_conformal.csv", index=False)
    json.dump({"Q_marginal": float(Qm), "Q_by_issue_bin": {CAT_NAMES[k]: float(v) for k, v in Qb.items()}, "Q_by_pred_bin": {CAT_NAMES[k]: float(v) for k, v in Qp.items()}},
              open(f"{RES}/c6_conformal_q.json", "w"), indent=1)
    log("C6 done")


# ------------------------------------------------------------------ C7
def c7_unseen_cities(D1, k=5):
    """Grouped k-fold over cities: every test city is absent from training.
    city_mean_aqi is replaced by the mean over training cities, because an
    unseen city has no history in the training set."""
    X = D1[FEATURES].astype(float).copy()
    y = D1.target.values.astype(float)
    l0 = D1.lag0.values.astype(float)
    d = y - l0
    te = (D1.split == "test").values
    cities = np.array(sorted(D1.City.unique()))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(cities))
    folds = [cities[order[i::k]] for i in range(k)]
    rows = []
    preds = np.full(len(D1), np.nan)
    for i, fold in enumerate(folds):
        inf = D1.City.isin(fold).values
        fit = (~inf) & (~te)
        prd = inf & te
        Xf = X.copy()
        m_city = X.loc[fit, "city_mean_aqi"].mean()
        Xf.loc[inf, "city_mean_aqi"] = m_city
        pS = make("hgb").fit(Xf[fit], d[fit]).predict(Xf[prd])
        pE = make("hgb").fit(Xf[fit], d[fit], sample_weight=1.0 + LAM * (y[fit] > 300)).predict(Xf[prd])
        p = l0[prd] + np.where(l0[prd] >= TAU_, pE, pS)
        preds[prd] = p
        yy = y[prd]
        rows.append({"Fold": i + 1, "Cities": len(fold), "Test days": int(prd.sum()),
                     "Persistence RMSE": rmse(yy, l0[prd]), "RASE RMSE": rmse(yy, p),
                     "Gain (%)": 100 * (1 - rmse(yy, p) / rmse(yy, l0[prd])),
                     "RMSE >300": rmse(yy[yy > 300], p[yy > 300]) if (yy > 300).sum() > 5 else np.nan,
                     "Persistence RMSE >300": rmse(yy[yy > 300], l0[prd][yy > 300]) if (yy > 300).sum() > 5 else np.nan})
        log(f"   C7 fold {i+1} done")
    df = pd.DataFrame(rows)
    yy, pp, aa = y[te], preds[te], l0[te]
    ok = ~np.isnan(pp)
    df.loc[len(df)] = {"Fold": "Pooled", "Cities": len(cities), "Test days": int(ok.sum()),
                       "Persistence RMSE": rmse(yy[ok], aa[ok]), "RASE RMSE": rmse(yy[ok], pp[ok]),
                       "Gain (%)": 100 * (1 - rmse(yy[ok], pp[ok]) / rmse(yy[ok], aa[ok])),
                       "RMSE >300": rmse(yy[ok & (yy > 300)], pp[ok & (yy > 300)]),
                       "Persistence RMSE >300": rmse(yy[ok & (yy > 300)], aa[ok & (yy > 300)])}
    df.round(2).to_csv(f"{RES}/c7_unseen_cities.csv", index=False)
    log("C7 done")


if __name__ == "__main__":
    t0 = time.time()
    D1 = pd.read_pickle(f'{OUT}/D1.pkl')
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "c2c3"): c2_c3(D1)
    if which in ("all", "c7"): c7_unseen_cities(D1)
    if which in ("all", "c6"): c6_conformal(D1)
    if which in ("all", "c1c4"): c1_c4(D1)
    if which in ("all", "c5"): c5_protocol_b(D1)
    log(f"Stage C ({which}) total {(time.time()-t0)/60:.1f} min")
