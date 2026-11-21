"""Stage A - the main run: all learners, all baselines, three horizons.
Writes results/predictions_h{h}.csv.gz with one column per model.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core
from core import FEATURES, LABEL, make, SEED
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = os.environ.get("RASE_OUT", "results")
RES = OUT
os.makedirs(RES, exist_ok=True)
LAM, TAU_ = 4, 150


def fit_stream(X, d, w, tr, va, te, tag="", names=("rf", "et", "gbr", "hgb")):
    trva = tr | va
    V, T = {}, {}
    for n in names:
        s = time.time()
        V[n] = make(n).fit(X[tr], d[tr], sample_weight=None if w is None else w[tr]).predict(X[va])
        T[n] = make(n).fit(X[trva], d[trva], sample_weight=None if w is None else w[trva]).predict(X[te])
        print(f"   [{tag}] {LABEL[n]:24s} {time.time()-s:6.0f} s", flush=True)
    meta = LinearRegression(positive=True).fit(np.column_stack([V[n] for n in names]), d[va],
                                               sample_weight=None if w is None else w[va])
    Tm = np.column_stack([T[n] for n in names])
    return {"single": T, "stack": meta.predict(Tm), "avg": Tm.mean(1),
            "weights": dict(zip(names, meta.coef_.round(4))),
            "intercept": float(meta.intercept_)}


# ---------------------------------------------------------------- new baselines
def gbdt_baselines(X, d, tr, va, te):
    """XGBoost, LightGBM and CatBoost on the same change target."""
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostRegressor
    trva = tr | va
    out = {}
    Xf = X.fillna(X.median()).fillna(0)
    t = time.time()
    m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, n_jobs=2, random_state=SEED,
                         tree_method="hist")
    m.fit(X[trva], d[trva]); out["XGBoost"] = m.predict(X[te]); print(f"   XGBoost {time.time()-t:.0f}s", flush=True)
    t = time.time()
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=48, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          n_jobs=2, random_state=SEED, verbose=-1)
    m.fit(X[trva], d[trva]); out["LightGBM"] = m.predict(X[te]); print(f"   LightGBM {time.time()-t:.0f}s", flush=True)
    t = time.time()
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                          random_seed=SEED, verbose=0, thread_count=2)
    m.fit(Xf[trva], d[trva]); out["CatBoost"] = m.predict(Xf[te]); print(f"   CatBoost {time.time()-t:.0f}s", flush=True)
    return out


def ts_baselines(data, h, tr, va, te):
    """Classical time-series baselines, computed per city.
    climatology: city x day-of-year mean over the training span.
    holt       : damped-trend exponential smoothing, h-step ahead at every origin.
    ar6        : AR(6) on AQI levels fitted per city on train+val, iterated h steps.
    All use information available at the forecast issue date only.
    """
    trva = (tr | va)
    out = {k: np.full(te.sum(), np.nan) for k in ("Climatology", "Exponential smoothing", "AR(6)")}
    te_idx = np.where(te)[0]
    pos = {ix: j for j, ix in enumerate(te_idx)}
    d_tr = data.loc[trva]
    # climatology: mean AQI by city and day-of-year of the TARGET date, training span only
    doy_tr = d_tr.TargetDate.dt.dayofyear
    clim = d_tr.assign(doy=doy_tr).groupby(["City", "doy"]).target.mean()
    city_mean = d_tr.groupby("City").target.mean()
    sub = data.loc[te]
    key = list(zip(sub.City, sub.TargetDate.dt.dayofyear))
    cvals = clim.reindex(pd.MultiIndex.from_tuples(key)).values
    fallback = sub.City.map(city_mean).values
    out["Climatology"] = np.where(np.isnan(cvals), fallback, cvals)

    # Holt damped trend and AR(6), per city, on the contiguous daily AQI series
    lagcols = [f"lag{k}" for k in range(7)]
    for city, g in data.groupby("City"):
        g = g.sort_values("Date")
        gi = g.index.values
        is_te = te[gi]
        # --- Holt damped trend on lag0 history, recursion over this city's dates
        a = g.lag0.values.astype(float)
        alpha, beta, phi = 0.4, 0.1, 0.9
        lvl = a[0]; trd = 0.0
        hf = np.empty(len(a))
        for i in range(len(a)):
            # forecast h steps ahead from the state BEFORE seeing day i+1
            damp = sum(phi ** (k + 1) for k in range(h))
            hf[i] = lvl + damp * trd
            if not np.isnan(a[i]):
                prev = lvl
                lvl = alpha * a[i] + (1 - alpha) * (lvl + phi * trd)
                trd = beta * (lvl - prev) + (1 - beta) * phi * trd
        m = is_te
        if m.any():
            for ix, v in zip(gi[m], hf[m]):
                out["Exponential smoothing"][pos[ix]] = v
        # --- AR(6) on levels, OLS on train+val rows of this city, iterated h steps
        gm = trva[gi]
        Z = g.loc[:, lagcols].values.astype(float)
        yv = g.lag0.values.astype(float)
        ok = gm & ~np.isnan(Z).any(1) & ~np.isnan(yv)
        if ok.sum() > 200:
            # regress AQI(t) on AQI(t-1) ... AQI(t-6), i.e. lag1..lag6 predicting lag0
            Zt = Z[ok]
            A = np.column_stack([np.ones(ok.sum()), Zt[:, 1], Zt[:, 2], Zt[:, 3], Zt[:, 4], Zt[:, 5], Zt[:, 6]])
            b, *_ = np.linalg.lstsq(A, yv[ok], rcond=None)
            # iterate: state = [lag0..lag5] -> predict next
            Zte = Z[is_te]
            hist = [Zte[:, k] for k in range(6)]  # lag0..lag5 at issue time
            cur = None
            for step in range(h):
                xs = np.column_stack([np.ones(len(Zte))] + hist[:6])
                cur = xs @ b
                hist = [cur] + hist[:5]
            vals = cur
        else:
            vals = np.full(int(is_te.sum()), np.nan)
        for ix, v in zip(gi[is_te], vals):
            out["AR(6)"][pos[ix]] = v
    # any remaining gaps fall back to persistence
    pers = data.loc[te, "lag0"].values.astype(float)
    for k in out:
        v = np.asarray(out[k], float)
        out[k] = np.where(np.isnan(v), pers, v)
    return out


def run_horizon(full, h):
    print(f"\n===== Horizon h = {h} day(s) =====", flush=True)
    data = core.build_dataset(full, h)
    X = data[FEATURES].astype(float)
    y = data.target.values.astype(float)
    l0 = data.lag0.values.astype(float)
    d = y - l0
    tr, va, te = [(data.split == s).values for s in ("train", "val", "test")]
    trva = tr | va
    print(f" train {tr.sum():,} | validation {va.sum():,} | test {te.sum():,}", flush=True)
    S = fit_stream(X, d, None, tr, va, te, "standard")
    print("   standard-stack weights:", S["weights"], flush=True)
    E = fit_stream(X, d, 1.0 + LAM * (y > 300), tr, va, te, "episode")
    print("   episode-stack weights:", E["weights"], flush=True)
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          Ridge(alpha=1.0)).fit(X[trva], d[trva])
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True,
                                     max_iter=200, random_state=SEED)).fit(X[trva], d[trva])
    a = l0[te]
    P = data.loc[te, ["Date", "TargetDate", "City"]].reset_index(drop=True)
    P["y"] = y[te]
    P["Persistence"] = a
    for k, v in ts_baselines(data, h, tr, va, te).items():
        P[k] = v
    P["Ridge regression"] = a + ridge.predict(X[te])
    P["Multilayer perceptron"] = a + mlp.predict(X[te])
    for k, v in S["single"].items():
        P[LABEL[k]] = a + v
    for k, v in gbdt_baselines(X, d, tr, va, te).items():
        P[k] = a + v
    P["Average ensemble"] = a + S["avg"]
    P["Standard stack"] = a + S["stack"]
    P["Episode stack"] = a + E["stack"]
    P["RASE (proposed)"] = np.where(a >= TAU_, P["Episode stack"], P["Standard stack"])
    P.to_csv(f"{RES}/predictions_h{h}.csv.gz", index=False)
    json.dump({"standard_weights": {k: float(v) for k, v in S["weights"].items()},
               "episode_weights": {k: float(v) for k, v in E["weights"].items()},
               "standard_intercept": S["intercept"], "episode_intercept": E["intercept"],
               "gate_share_pct": float(100 * np.mean(a >= TAU_)),
               "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
               "n_total": int(len(data))},
              open(f"{RES}/run_h{h}.json", "w"), indent=1)
    print(f" gate routed {100*np.mean(a >= TAU_):.1f}% of test days to the episode stream", flush=True)
    del data, X
    return P


if __name__ == "__main__":
    t0 = time.time()
    full = pd.read_pickle(f'{OUT}/full.pkl')
    full = full[full.Date <= core.TEST_END]
    for h in (1, 2, 3):
        run_horizon(full, h)
    print(f"\nStage A total {(time.time()-t0)/60:.1f} min", flush=True)
