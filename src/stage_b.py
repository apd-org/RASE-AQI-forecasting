"""Stage B - every table that follows from the saved predictions.
No model fitting here, so each number traces back to one run.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core
from core import metrics, aqi_category, CAT_NAMES, dm_hac, holm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT = os.environ.get("RASE_OUT", "results")
RES = OUT
TAB = OUT
os.makedirs(TAB, exist_ok=True)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, np.clip(b, 0, 500))))

ORDER = ["Persistence", "Climatology", "Exponential smoothing", "AR(6)", "Ridge regression",
         "Multilayer perceptron", "Random Forest", "Extra Trees", "Gradient Boosting",
         "Hist. Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Average ensemble",
         "Standard stack", "Episode stack", "RASE (proposed)"]


def load(h):
    P = pd.read_csv(f"{RES}/predictions_h{h}.csv.gz", parse_dates=["Date", "TargetDate"])
    return P


def b1_main(Ps):
    rows = []
    for h, P in Ps.items():
        for m in ORDER:
            if m not in P.columns:
                continue
            d = metrics(P.y, P[m])
            d.update({"Horizon": h, "Model": m})
            rows.append(d)
    T = pd.DataFrame(rows).set_index(["Horizon", "Model"])
    T.round(3).to_csv(f"{TAB}/b1_main_multihorizon.csv")
    return T


def b2_categories(Ps, models=("Persistence", "Standard stack", "CatBoost", "RASE (proposed)")):
    rows = []
    for h, P in Ps.items():
        cy = aqi_category(P.y.values)
        for b, nm in enumerate(CAT_NAMES):
            m = cy == b
            if m.sum() == 0:
                continue
            rec = {"Horizon": h, "Category": nm, "n": int(m.sum())}
            for mod in models:
                if mod not in P.columns:
                    continue
                p = np.clip(P[mod].values, 0, 500)
                rec[f"{mod} RMSE"] = rmse(P.y.values[m], p[m])
                rec[f"{mod} MAE"] = float(mean_absolute_error(P.y.values[m], p[m]))
                rec[f"{mod} bias"] = float(np.mean(p[m] - P.y.values[m]))
            rows.append(rec)
    T = pd.DataFrame(rows)
    T.round(2).to_csv(f"{TAB}/b2_categories.csv", index=False)
    return T


def b3_dm(Ps):
    rows = []
    for h, P in Ps.items():
        models = [m for m in ORDER if m in P.columns and m != "RASE (proposed)"]
        for subset, nm in ((None, "All days"), ((P.y > 200).values, "AQI > 200"),
                           ((P.y > 300).values, "AQI > 300")):
            recs = []
            for m in models:
                r = dm_hac(P, m, "RASE (proposed)", h, subset=subset)
                r.update({"Horizon": h, "Subset": nm, "Comparison": f"{m} vs RASE"})
                recs.append(r)
            p = [r["p-value"] for r in recs]
            adj = holm(p)
            for r, a in zip(recs, adj):
                r["p (Holm)"] = float(a)
                rows.append(r)
    T = pd.DataFrame(rows)[["Horizon", "Subset", "Comparison", "DM statistic", "p-value",
                            "p (Holm)", "mean loss diff", "CI low", "CI high", "HAC lag", "n days"]]
    T.round(4).to_csv(f"{TAB}/b3_dm_tests.csv", index=False)
    return T


def b4_cities(Ps):
    P = Ps[1]
    rows = []
    singles = [m for m in ("Random Forest", "Extra Trees", "Gradient Boosting",
                           "Hist. Gradient Boosting") if m in P.columns]
    for c, g in P.groupby("City"):
        rows.append({"City": c, "Test days": len(g),
                     "Mean AQI": float(g.y.mean()),
                     "Days AQI>300": int((g.y > 300).sum()),
                     "Persistence": rmse(g.y, g["Persistence"]),
                     "Best single learner": min(rmse(g.y, g[m]) for m in singles),
                     "Standard stack": rmse(g.y, g["Standard stack"]),
                     "RASE": rmse(g.y, g["RASE (proposed)"]),
                     "RASE R2": float(r2_score(g.y, np.clip(g["RASE (proposed)"], 0, 500)))})
    S = pd.DataFrame(rows).set_index("City")
    S["Reduction vs persistence (%)"] = 100 * (1 - S.RASE / S.Persistence)
    S.round(3).to_csv(f"{TAB}/b4_cities.csv")
    return S


def b5_warnings(Ps, thresholds=(200, 300)):
    """Contingency skill for issuing a warning: the forecast crosses the
    threshold when the observation does."""
    rows = []
    for h, P in Ps.items():
        for thr in thresholds:
            obs = (P.y.values > thr)
            for m in ORDER:
                if m not in P.columns:
                    continue
                fc = (np.clip(P[m].values, 0, 500) > thr)
                H = int((fc & obs).sum()); M = int((~fc & obs).sum())
                F = int((fc & ~obs).sum())
                pod = H / (H + M) if H + M else np.nan
                far = F / (H + F) if H + F else np.nan
                csi = H / (H + M + F) if H + M + F else np.nan
                rows.append({"Horizon": h, "Threshold": thr, "Model": m, "Hits": H,
                             "Misses": M, "False alarms": F, "POD": pod, "FAR": far, "CSI": csi})
    T = pd.DataFrame(rows)
    T.round(3).to_csv(f"{TAB}/b5_warning_skill.csv", index=False)
    return T


def b6_components(Ps):
    """Isolates the three ingredients of RASE from the saved columns."""
    rows = []
    for h, P in Ps.items():
        a = P["Persistence"].values.astype(float)
        singles = ["Random Forest", "Extra Trees", "Gradient Boosting", "Hist. Gradient Boosting"]
        best = min(singles, key=lambda m: rmse(P.y, P[m]))
        comp = {"Persistence": a,
                f"Change target, best single learner ({best})": P[best].values,
                "Stacking only (standard stack)": P["Standard stack"].values,
                "Episode weighting only (episode stack)": P["Episode stack"].values,
                "Stacking + weighting + gate (RASE)": P["RASE (proposed)"].values}
        for nm, p in comp.items():
            y = P.y.values
            rows.append({"Horizon": h, "Configuration": nm, "RMSE": rmse(y, p),
                         "MAE": float(mean_absolute_error(y, np.clip(p, 0, 500))),
                         "RMSE >200": rmse(y[y > 200], p[y > 200]),
                         "RMSE >300": rmse(y[y > 300], p[y > 300])})
    T = pd.DataFrame(rows)
    T.round(2).to_csv(f"{TAB}/b6_components.csv", index=False)
    return T


def b7_protocols(Ps):
    """Protocol A (published) against Protocol B (out-of-fold meta-learner,
    design year separated from the calibration year)."""
    f = f"{RES}/c5_protocol_b_meta.json"
    if not os.path.exists(f):
        return None
    meta = json.load(open(f))
    P = Ps[1]
    names = ["Random Forest", "Extra Trees", "Gradient Boosting", "Hist. Gradient Boosting"]
    a = P["Persistence"].values.astype(float)
    base_s = np.column_stack([P[n].values - a for n in names])
    epi_file = f"{TAB}/epi_base_test.npy"
    if not os.path.exists(epi_file):
        print("episode-stream base predictions missing; skipping protocol comparison")
        return None
    base_e = np.load(epi_file)
    cS, iS = np.asarray(meta["std"]["coef"]), meta["std"]["intercept"]
    cE, iE = np.asarray(meta["epi"]["coef"]), meta["epi"]["intercept"]
    pS = a + base_s @ cS + iS
    pE = a + base_e @ cE + iE
    tau = meta.get("tau_selected_on_2023", 150)
    pB = np.where(a >= tau, pE, pS)
    y = P.y.values
    rows = []
    for nm, p in (("Protocol A: meta on 2024, tau = 150 (as published)", P["RASE (proposed)"].values),
                  (f"Protocol B: out-of-fold meta, tau = {tau} chosen on 2023", pB)):
        rows.append({"Protocol": nm, "RMSE": rmse(y, p),
                     "MAE": float(mean_absolute_error(y, np.clip(p, 0, 500))),
                     "R2": float(r2_score(y, np.clip(p, 0, 500))),
                     "RMSE >200": rmse(y[y > 200], p[y > 200]),
                     "RMSE >300": rmse(y[y > 300], p[y > 300])})
    T = pd.DataFrame(rows)
    T.round(3).to_csv(f"{TAB}/b7_protocols.csv", index=False)
    # DM between the two protocols on severe days
    Q = P.copy(); Q["Protocol B"] = pB
    r = dm_hac(Q, "Protocol B", "RASE (proposed)", 1, subset=(Q.y > 300).values)
    json.dump(r, open(f"{TAB}/b7_protocol_dm.json", "w"), indent=1)
    return T


if __name__ == "__main__":
    Ps = {h: load(h) for h in (1, 2, 3) if os.path.exists(f"{RES}/predictions_h{h}.csv.gz")}
    print("horizons loaded:", list(Ps))
    T1 = b1_main(Ps); print(T1.loc[1].round(2).to_string())
    b2_categories(Ps); b3_dm(Ps); b4_cities(Ps); b5_warnings(Ps); b6_components(Ps); b7_protocols(Ps)
    print("Stage B done")
