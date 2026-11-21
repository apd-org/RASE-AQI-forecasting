"""Stage D - exact Shapley values for the complete gated ensemble.

Because the meta-learner is linear with non-negative coefficients and the gate
selects one stream per city-day, the Shapley value of the whole forecast is the
coefficient-weighted sum of the learners' exact TreeSHAP values for the stream
the gate selected.  No sampling is involved, so the decomposition is exact and
additivity can be verified to machine precision.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core
from core import FEATURES, LABEL, make, SEED
import shap

OUT = os.environ.get("RASE_OUT", "results")
RES = OUT
TAB = OUT
os.makedirs(TAB, exist_ok=True)
LAM, TAU_ = 4, 150
N_SEV, N_REST = 160, 160
NAMES = ("rf", "et", "gbr", "hgb")

PRETTY = {
 "lag0": "AQI today (t)", "lag1": "AQI at t-1", "lag2": "AQI at t-2", "lag3": "AQI at t-3",
 "lag4": "AQI at t-4", "lag5": "AQI at t-5", "lag6": "AQI at t-6", "lag13": "AQI at t-13",
 "lag_1y": "AQI one year ago", "roll_mean3": "3-day mean", "roll_mean7": "7-day mean",
 "roll_mean14": "14-day mean", "roll_mean30": "30-day mean", "roll_std7": "7-day s.d.",
 "roll_max7": "7-day maximum", "diff1": "Change since t-1", "diff7": "Change since t-7",
 "nat_mean": "National mean AQI", "igp_mean": "IGP mean AQI", "igp_mean_diff1": "IGP mean change",
 "city_minus_igp": "City minus IGP mean", "is_igp": "City in IGP", "city_mean_aqi": "City climatology",
 "pol_pm25": "PM2.5 lead pollutant", "pol_pm10": "PM10 lead pollutant", "pol_gas": "Gas lead pollutant",
 "obs_days7": "Observed days in last 7", "doy_sin": "Season (sin)", "doy_cos": "Season (cos)",
 "month": "Month", "weekday": "Weekday", "stubble_season": "Oct-Nov stubble season"}


def fitted_streams(D1):
    X = D1[FEATURES].astype(float)
    y = D1.target.values.astype(float)
    l0 = D1.lag0.values.astype(float)
    d = y - l0
    tr, va, te = [(D1.split == s).values for s in ("train", "val", "test")]
    trva = tr | va
    models = {"std": {}, "epi": {}}
    w = 1.0 + LAM * (y > 300)
    for stream, sw in (("std", None), ("epi", w)):
        for n in NAMES:
            t = time.time()
            models[stream][n] = make(n).fit(X[trva], d[trva],
                                            sample_weight=None if sw is None else sw[trva])
            print(f"   refit {stream} {n} {time.time()-t:.0f}s", flush=True)
    return models, X, y, l0, te


def main():
    D1 = pd.read_pickle(f'{OUT}/D1.pkl')
    meta = json.load(open(f"{RES}/run_h1.json"))
    import pickle
    cache = f"{OUT}/models_h1.pkl"
    if os.path.exists(cache):
        models, X, y, l0, te = pickle.load(open(cache, "rb"))
        print("loaded cached models", flush=True)
    else:
        models, X, y, l0, te = fitted_streams(D1)
        pickle.dump((models, X, y, l0, te), open(cache, "wb"))

    # verify the refit reproduces the saved predictions
    P = pd.read_csv(f"{RES}/predictions_h1.csv.gz")
    Xte = X[te]
    chk = {}
    for n in NAMES:
        chk[n] = float(np.abs(l0[te] + models["std"][n].predict(Xte) - P[LABEL[n]].values).max())
    print("refit vs saved, max abs deviation:", chk, flush=True)

    wS = np.array([meta["standard_weights"][n] for n in NAMES])
    wE = np.array([meta["episode_weights"][n] for n in NAMES])
    iS, iE = meta["standard_intercept"], meta["episode_intercept"]

    # explain a stratified sample: every Very Poor / Severe day plus a random slice
    sub = P.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sev_all = np.where(sub.y.values > 300)[0]
    sev = rng.choice(sev_all, size=min(N_SEV, len(sev_all)), replace=False)
    rest = rng.choice(np.setdiff1d(np.arange(len(sub)), sev_all), size=N_REST, replace=False)
    idx = np.sort(np.concatenate([sev, rest]))
    Xs = Xte.iloc[idx]
    a_s = l0[te][idx]
    gate = a_s >= TAU_

    SV = {}
    for stream in ("std", "epi"):
        acc = np.zeros((len(idx), len(FEATURES)))
        base = 0.0
        ws = wS if stream == "std" else wE
        for k, n in enumerate(NAMES):
            m = models[stream][n]
            inner = m.est_ if hasattr(m, "est_") else m
            Xin = Xs.fillna(m.med_).fillna(0) if hasattr(m, "est_") else Xs
            ex = shap.TreeExplainer(inner)
            sv = ex.shap_values(Xin, check_additivity=False)
            acc += ws[k] * sv
            base += ws[k] * float(np.atleast_1d(ex.expected_value)[0])
            print(f"   shap {stream} {n} done", flush=True)
        SV[stream] = (acc, base + (iS if stream == "std" else iE))

    phi = np.where(gate[:, None], SV["epi"][0], SV["std"][0])
    base = np.where(gate, SV["epi"][1], SV["std"][1])
    delta_hat = np.where(gate,
                         sub["Episode stack"].values[idx] - a_s,
                         sub["Standard stack"].values[idx] - a_s)
    add_err = float(np.abs(phi.sum(1) + base - delta_hat).max())
    print("exact-Shapley additivity, max abs error:", add_err, flush=True)

    np.save(f"{TAB}/d_shap_values.npy", phi)
    np.save(f"{TAB}/d_shap_index.npy", idx)
    np.save(f"{TAB}/d_shap_base.npy", base)
    Xs.to_pickle(f"{TAB}/d_shap_X.pkl")

    imp = pd.Series(np.abs(phi).mean(0), index=[PRETTY.get(f, f) for f in FEATURES])
    imp.sort_values(ascending=False).round(4).to_csv(f"{TAB}/d_global_importance.csv")
    sev_mask = sub.y.values[idx] > 300
    comp = pd.DataFrame({"All explained days": np.abs(phi).mean(0),
                         "Very Poor / Severe days": np.abs(phi[sev_mask]).mean(0)},
                        index=[PRETTY.get(f, f) for f in FEATURES])
    comp.sort_values("All explained days", ascending=False).round(4).to_csv(f"{TAB}/d_importance_split.csv")
    json.dump({"additivity_max_abs_error": add_err,
               "n_explained": int(len(idx)), "n_severe_explained": int(sev_mask.sum()),
               "refit_max_abs_deviation": chk},
              open(f"{TAB}/d_shap_meta.json", "w"), indent=1)
    print("Stage D done", flush=True)


if __name__ == "__main__":
    main()
