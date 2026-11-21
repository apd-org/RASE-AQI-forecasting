"""Verbatim re-use of the RASE v2 notebook's parsing, features and learners.
Kept byte-identical in logic so that every new experiment is comparable to the
authoritative 15-Nov run.  Data root is passed in, never inferred.
"""
import os, re, csv, glob, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

SEED = 42
STUDY_START, TRAIN_END, VAL_END, TEST_END = "2018-01-29", "2023-12-31", "2024-12-31", "2026-10-14"

CATS = {'good': (0, 50), 'satisfactory': (51, 100), 'moderate': (101, 200),
        'poor': (201, 300), 'very poor': (301, 400), 'severe': (401, 500)}

IGP = {"Delhi","Noida","Greater Noida","Ghaziabad","Gurugram","Faridabad","Meerut","Hapur",
"Bulandshahr","Baghpat","Muzaffarnagar","Moradabad","Bareilly","Agra","Firozabad","Mathura",
"Vrindavan","Kanpur","Lucknow","Prayagraj","Varanasi","Gorakhpur","Patna","Gaya","Muzaffarpur",
"Hajipur","Panipat","Sonipat","Rohtak","Bhiwani","Jind","Hisar","Fatehabad","Kaithal","Karnal",
"Kurukshetra","Yamuna Nagar","Ambala","Bahadurgarh","Ballabgarh","Dharuhera","Manesar",
"Charkhi Dadri","Narnaul","Palwal","Amritsar","Ludhiana","Jalandhar","Patiala","Bathinda",
"Khanna","Mandi Gobindgarh","Rupnagar","Chandigarh","Kolkata","Howrah","Asansol","Durgapur",
"Siliguri"}

dpat = re.compile(r"(20\d\d)-?(\d\d)-?(\d\d)$")


def files(root):
    fs = sorted(glob.glob(f"{root}/data/Data_upto_20231218/*.csv")) + \
         [f"{root}/data/20231201_20250304/AQI.csv", f"{root}/AQI.csv"]
    return fs


def parse(root):
    """Returns the cleaned frame plus a full audit trail of every exclusion."""
    rows = []
    audit = dict(raw=0, no_category=0, no_date=0, no_city=0, unparsable=0, out_of_range=0)
    for f in files(root):
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf8", errors="ignore") as fh:
            for r in csv.reader(fh):
                r = [x.strip() for x in r]
                if not any(r) or "Air Quality" in r:
                    if r and r[-1].startswith("Date") and len(r[-1]) > 4:
                        r = [r[-1][4:]] + r
                    else:
                        continue
                audit['raw'] += 1
                cat = next((x.lower() for x in r if x.lower() in CATS), None)
                date = None
                for x in r[::-1]:
                    m = dpat.match(x)
                    if m:
                        date = f"{m[1]}-{m[2]}-{m[3]}"
                        break
                vals = [x for x in r if x]
                city = vals[0] if vals else None
                if r[0] == "" and len(r) > 1:
                    city = r[1]
                try:
                    i = r.index(next(x for x in r if x.lower() == cat))
                    aqi = float(r[i + 1])
                    pol = r[i + 2] if len(r) > i + 2 else ""
                except Exception:
                    audit['unparsable'] += 1
                    continue
                if cat is None:
                    audit['no_category'] += 1; continue
                if date is None:
                    audit['no_date'] += 1; continue
                if not city:
                    audit['no_city'] += 1; continue
                lo, hi = CATS[cat]
                if not (lo <= aqi <= hi):
                    audit['out_of_range'] += 1; continue
                rows.append((city, date, aqi, pol))
    df = pd.DataFrame(rows, columns=["city", "date", "aqi", "pollutant"])
    audit['kept_before_dedup'] = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n0 = len(df); df = df.dropna(subset=["date"]); audit['bad_date_dropped'] = n0 - len(df)
    key = df["city"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    canon = df.assign(k=key).groupby("k")["city"].agg(lambda s: s.value_counts().index[0])
    audit['distinct_city_strings'] = int(df['city'].nunique())
    df["city"] = key.map(canon)
    audit['distinct_cities_after_merge'] = int(df['city'].nunique())
    n0 = len(df); df = df.drop_duplicates(["city", "date"], keep="last")
    audit['duplicate_city_days_dropped'] = n0 - len(df)
    audit['valid_city_days'] = len(df)
    return df, audit


def pollutant_flags(s):
    s = s.fillna("").astype(str)
    pm25 = s.str.contains("2.5", regex=False)
    pm10 = s.str.contains("10", regex=False)
    gas = s.str.contains(r"CO|NO|SO|O3|NH", regex=True) & ~(pm25 | pm10)
    return pd.DataFrame({"pol_pm25": pm25.astype(int), "pol_pm10": pm10.astype(int),
                         "pol_gas": gas.astype(int)})


GROUPS = {
 "Recent history": ["lag0","lag1","lag2","lag3","lag4","lag5","lag6","lag13","lag_1y"],
 "Trends": ["roll_mean3","roll_mean7","roll_mean14","roll_mean30","roll_std7","roll_max7","diff1","diff7"],
 "Regional": ["nat_mean","igp_mean","igp_mean_diff1","city_minus_igp"],
 "City context": ["is_igp","city_mean_aqi","pol_pm25","pol_pm10","pol_gas","obs_days7"],
 "Calendar": ["doy_sin","doy_cos","month","weekday","stubble_season"]}
FEATURES = sum(GROUPS.values(), [])


def build_dataset(full, h=1, min_days=2500):
    full = full[full.Date >= STUDY_START]
    counts = full.groupby("City").size(); cities = counts[counts >= min_days].index
    d = full[full.City.isin(cities)].reset_index(drop=True)
    d = pd.concat([d, pollutant_flags(d["Pollutant"])], axis=1)
    nat = full.groupby("Date").AQI.mean().rename("nat_mean")
    igp = full[full.City.isin(IGP)].groupby("Date").AQI.mean().rename("igp_mean")
    reg = pd.concat([nat, igp], axis=1)
    reg = reg.reindex(pd.date_range(reg.index.min(), reg.index.max(), freq="D"))
    out = []; dates = pd.date_range(STUDY_START, d.Date.max(), freq="D")
    for city, g in d.groupby("City"):
        g = g.set_index("Date").reindex(dates); a = g["AQI"]
        f = pd.DataFrame(index=dates); f["City"] = city
        f["target"] = a.shift(-h)
        for k in range(7): f[f"lag{k}"] = a.shift(k)
        f["lag13"] = a.shift(13); f["lag_1y"] = a.shift(365 - h)
        for w in (3, 7, 14, 30):
            f[f"roll_mean{w}"] = a.rolling(w, min_periods=max(2, w // 2)).mean()
        f["roll_std7"] = a.rolling(7, min_periods=4).std()
        f["roll_max7"] = a.rolling(7, min_periods=4).max()
        f["diff1"] = a - a.shift(1); f["diff7"] = a - a.shift(7)
        f["obs_days7"] = a.notna().rolling(7, min_periods=1).sum()
        for c in ("pol_pm25", "pol_pm10", "pol_gas"): f[c] = g[c]
        f["nat_mean"] = reg["nat_mean"].reindex(dates).values
        f["igp_mean"] = reg["igp_mean"].reindex(dates).values
        f["igp_mean_diff1"] = f["igp_mean"] - reg["igp_mean"].shift(1).reindex(dates).values
        f["city_minus_igp"] = f["lag0"] - f["igp_mean"]; f["is_igp"] = int(city in IGP)
        t = dates + pd.Timedelta(days=h); doy = t.dayofyear
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        f["month"] = t.month; f["weekday"] = t.weekday
        f["stubble_season"] = ((t.month == 10) | (t.month == 11)).astype(int)
        out.append(f)
    data = pd.concat(out).rename_axis("Date").reset_index().dropna(subset=["target", "lag0"])
    data["TargetDate"] = data.Date + pd.Timedelta(days=h)
    data = data[data.TargetDate <= TEST_END]
    data["split"] = np.where(data.TargetDate <= TRAIN_END, "train",
                             np.where(data.TargetDate <= VAL_END, "val", "test"))
    enc = data[data.split == "train"].groupby("City").lag0.mean().rename("city_mean_aqi")
    return data.merge(enc, left_on="City", right_index=True, how="left").reset_index(drop=True)


class MedianFill(BaseEstimator, RegressorMixin):
    def __init__(self, est=None): self.est = est
    def fit(self, X, y, sample_weight=None):
        X = pd.DataFrame(X); self.med_ = X.median()
        self.est_ = clone(self.est).fit(X.fillna(self.med_).fillna(0), y, sample_weight=sample_weight)
        return self
    def predict(self, X):
        return self.est_.predict(pd.DataFrame(X).fillna(self.med_).fillna(0))


def make(name):
    if name == "rf":
        return RandomForestRegressor(n_estimators=120, min_samples_leaf=5, max_features=0.5,
                                     max_samples=0.4, n_jobs=-1, random_state=SEED)
    if name == "et":
        return MedianFill(ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                              max_features=0.5, n_jobs=-1, random_state=SEED))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=48,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             random_state=SEED)
    if name == "gbr":
        return MedianFill(GradientBoostingRegressor(n_estimators=220, max_depth=5,
                                                    learning_rate=0.06, subsample=0.4,
                                                    max_features=0.5, random_state=SEED))
    raise KeyError(name)


LABEL = {"rf": "Random Forest", "et": "Extra Trees", "gbr": "Gradient Boosting",
         "hgb": "Hist. Gradient Boosting"}

CAT_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def aqi_category(a):
    return np.digitize(np.asarray(a), [50.5, 100.5, 200.5, 300.5, 400.5])


def metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 500)
    cy, cp = aqi_category(y), aqi_category(p)
    r = lambda m: float(np.sqrt(mean_squared_error(y[m], p[m]))) if m.any() else np.nan
    return {"RMSE": r(y >= 0), "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p),
            "MAPE (%)": 100 * np.mean(np.abs(y - p) / np.maximum(y, 1)),
            "Category acc. (%)": 100 * np.mean(cy == cp),
            "Within one category (%)": 100 * np.mean(np.abs(cy - cp) <= 1),
            "RMSE AQI>200": r(y > 200), "RMSE AQI>300": r(y > 300)}


def newey_west_var(x, lag):
    x = np.asarray(x, float) - np.mean(x); T = len(x); v = np.dot(x, x) / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * np.dot(x[k:], x[:-k]) / T
    return v


def dm_hac(P, a, b, h, subset=None):
    """Diebold-Mariano test on squared-error differences, aggregated by target
    date so that same-day dependence across cities is absorbed, and with a
    Newey-West HAC variance plus Harvey's small-sample correction for the
    MA(h-1) structure of h-step forecast errors.

    Days are weighted by the number of qualifying city-days, so the estimated
    mean loss differential equals the difference in pooled mean squared error
    over the same subset; the test and the reported RMSE therefore always agree
    in sign.  Positive statistic => b better than a.
    """
    Q = P if subset is None else P[subset]
    dif = ((Q.y - Q[a].clip(0, 500)) ** 2 - (Q.y - Q[b].clip(0, 500)) ** 2)
    g = dif.groupby(Q.TargetDate)
    s = g.sum().sort_index()          # daily sum of loss differences
    m = g.size().reindex(s.index)     # city-days contributing that day
    T = len(s)
    mu = float(s.sum() / m.sum())     # = MSE(a) - MSE(b) over the subset
    u = (s.values - mu * m.values) / m.values.mean()
    lag = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    var = newey_west_var(u, lag) / T
    harvey = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    se = np.sqrt(var) / harvey
    stat = mu / se if se > 0 else np.nan
    tcrit = stats.t.ppf(0.975, df=T - 1)
    return {"DM statistic": float(stat),
            "p-value": float(2 * (1 - stats.t.cdf(abs(stat), df=T - 1))),
            "mean loss diff": float(mu),
            "CI low": float(mu - tcrit * se),
            "CI high": float(mu + tcrit * se),
            "HAC lag": int(lag), "n days": int(T)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def load_full(root):
    df, audit = parse(root)
    full = df.rename(columns={"city": "City", "date": "Date", "aqi": "AQI",
                              "pollutant": "Pollutant"})
    return full, audit
