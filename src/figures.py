"""All figures, rendered at 600 dpi with large, readable axis labels."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import core
from core import CAT_NAMES, aqi_category

OUT = os.environ.get("RASE_OUT", "results")
RES = OUT
TAB = OUT
FIG = OUT
os.makedirs(FIG, exist_ok=True)
DPI = 600

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14,
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.linewidth": 0.9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "figure.facecolor": "white"})

RED, BLUE, GREY, GREEN, ORANGE = "#B22222", "#2A5CA5", "#8A8A8A", "#1F6354", "#C77B19"


def fig1_workflow(audit, meta):
    fig, ax = plt.subplots(figsize=(15.5, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off"); ax.grid(False)
    boxes = [
        ("CPCB daily AQI\nbulletins",
         f"Jun 2015 - 14 Oct 2026\n{audit['raw']:,} raw rows\n{audit['valid_city_days']:,} valid city-days",
         "#DCE8F7"),
        ("Cleaning and\nselection",
         f"Category-range check,\nname merge, de-duplication\n-> 63 cities, 2018-2026", "#DFEFDC"),
        ("Feature\nengineering",
         "32 features: lags, rolling\nstatistics, regional, city\ncontext, calendar;\ntarget = change in AQI", "#FBECD4"),
        ("Regime-aware\nstacking",
         "Standard stream +\nepisode stream (weight\n1+4 on AQI > 300);\ngate: AQI(t) >= 150", "#F3E0F2"),
        ("Output and\nevaluation",
         f"AQI at t+1, t+2, t+3\nTest {meta['n_test']:,} city-days\nExact Shapley values,\n80% conformal intervals", "#E6E6E6"),
    ]
    w, gap = 17.4, 2.6
    for i, (title, body, col) in enumerate(boxes):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 2), w, 26, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    linewidth=1.4, edgecolor="#333333", facecolor=col))
        ax.text(x + w / 2, 24.0, title, ha="center", va="center", fontsize=14.5, fontweight="bold")
        ax.text(x + w / 2, 12.5, body, ha="center", va="center", fontsize=11.0, linespacing=1.5)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.3, 15), (x + w + gap - 0.3, 15),
                                         arrowstyle="-|>", mutation_scale=18, linewidth=1.8,
                                         color="#333333"))
    fig.savefig(f"{FIG}/fig1_workflow.png", dpi=DPI)
    plt.close(fig)


def fig2_models(T1, Ps):
    order_all = T1.sort_values("RMSE")
    order_sev = T1.sort_values("RMSE AQI>300")
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))

    def colr(idx):
        return [RED if m.startswith("RASE") else GREY if m == "Persistence" else BLUE for m in idx]

    b0 = ax[0].barh(order_all.index, order_all.RMSE, color=colr(order_all.index), height=0.72)
    ax[0].set_xlim(0, order_all.RMSE.max() * 1.20)
    ax[0].bar_label(b0, fmt="%.2f", padding=3, fontsize=10.5)
    ax[0].set_xlabel("RMSE on all test days (AQI points)")
    ax[0].set_title("(a) Overall accuracy")
    b1 = ax[1].barh(order_sev.index, order_sev["RMSE AQI>300"], color=colr(order_sev.index), height=0.72)
    ax[1].set_xlim(0, order_sev["RMSE AQI>300"].max() * 1.20)
    ax[1].bar_label(b1, fmt="%.1f", padding=3, fontsize=10.5)
    ax[1].axvline(T1.loc["Persistence", "RMSE AQI>300"], color=GREY, ls="--", lw=1.6)
    ax[1].set_xlabel("RMSE on Very Poor and Severe days, AQI > 300 (AQI points)")
    ax[1].set_title("(b) Accuracy on the most polluted days")
    hs = sorted(Ps)
    for m, mk, c in (("Persistence", "o", GREY), ("CatBoost", "s", BLUE),
                     ("Standard stack", "D", GREEN), ("RASE (proposed)", "^", RED)):
        v = [float(np.sqrt(np.mean((Ps[h].y - np.clip(Ps[h][m], 0, 500)) ** 2))) for h in hs]
        ax[2].plot(hs, v, marker=mk, color=c, lw=2.0, ms=9, label=m)
    ax[2].set_xticks(hs); ax[2].set_xlabel("Forecast horizon (days ahead)")
    ax[2].set_ylabel("RMSE (AQI points)"); ax[2].legend(frameon=False)
    ax[2].set_title("(c) Accuracy against horizon")
    for a in ax[:2]:
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_models.png", dpi=DPI)
    plt.close(fig)


def fig3_city_and_coverage(P, conf):
    fig, ax = plt.subplots(1, 2, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.65, 1]})
    k = (P[P.City == "Kanpur"].sort_values("TargetDate")
         .set_index("TargetDate").asfreq("D").reset_index())
    ax[0].plot(k.TargetDate, k.y, lw=1.4, color="#333333", label="Observed AQI")
    ax[0].plot(k.TargetDate, np.clip(k["RASE (proposed)"], 0, 500), lw=1.5, color=RED,
               label="RASE forecast, one day ahead")
    ax[0].axhspan(300, 500, color=ORANGE, alpha=0.12)
    ax[0].text(k.TargetDate.iloc[12], 420, "Very Poor and Severe", fontsize=11.5, color="#8a5a10")
    ax[0].set_ylabel("AQI"); ax[0].set_xlabel("Target date")
    ax[0].set_title("(a) Kanpur over the test period")
    ax[0].legend(frameon=False, loc="upper right")

    c = conf[conf.Subset.isin(CAT_NAMES)]
    cats = [x for x in CAT_NAMES if x in set(c.Subset)]
    pred_key = "Mondrian (by predicted category)"
    if pred_key not in set(c.Method):
        pred_key = "Mondrian (by issue-day category)"
    marg = [float(c[(c.Method == "Marginal") & (c.Subset == s)]["Coverage (%)"].iloc[0]) for s in cats]
    mond = [float(c[(c.Method == pred_key) & (c.Subset == s)]["Coverage (%)"].iloc[0]) for s in cats]
    x = np.arange(len(cats)); w = 0.38
    ax[1].bar(x - w / 2, marg, w, label="Marginal conformal", color=BLUE)
    ax[1].bar(x + w / 2, mond, w, label="Conformal, conditioned on predicted category", color=RED)
    ax[1].axhline(80, color="#333333", ls="--", lw=1.6)
    ax[1].text(len(cats) - 0.45, 81.5, "nominal 80%", fontsize=11.5, ha="right")
    ax[1].set_xticks(x); ax[1].set_xticklabels(cats, rotation=20, ha="right")
    ax[1].set_ylabel("Empirical coverage (%)"); ax[1].set_xlabel("Observed CPCB category")
    ax[1].set_ylim(0, 105); ax[1].legend(frameon=False, loc="lower left")
    ax[1].set_title("(b) Interval coverage by category")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_city_coverage.png", dpi=DPI)
    plt.close(fig)


def fig4_shap():
    phi = np.load(f"{TAB}/d_shap_values.npy")
    idx = np.load(f"{TAB}/d_shap_index.npy")
    Xs = pd.read_pickle(f"{TAB}/d_shap_X.pkl")
    comp = pd.read_csv(f"{TAB}/d_importance_split.csv", index_col=0)
    P = pd.read_csv(f"{RES}/predictions_h1.csv.gz")
    sev = P.y.values[idx] > 300
    top = comp.head(12).index[::-1]
    from stage_d import PRETTY
    pretty = [PRETTY.get(f, f) for f in Xs.columns]
    pos = {p: j for j, p in enumerate(pretty)}

    fig, ax = plt.subplots(1, 2, figsize=(19, 7.2))
    yv = np.arange(len(top)); w = 0.4
    ax[0].barh(yv - w / 2, [comp.loc[t, "All explained days"] for t in top], w,
               color=BLUE, label="All explained days")
    ax[0].barh(yv + w / 2, [comp.loc[t, "Very Poor / Severe days"] for t in top], w,
               color=RED, label="Very Poor and Severe days")
    ax[0].set_yticks(yv); ax[0].set_yticklabels(top)
    ax[0].set_xlabel("Mean |Shapley value| (AQI points)")
    ax[0].set_title("(a) Contribution of each feature to the forecast change")
    ax[0].legend(frameon=False); ax[0].grid(axis="y", visible=False)

    show = list(top[::-1])[:10][::-1]
    rng = np.random.default_rng(0)
    for j, t in enumerate(show):
        col = pos[t]
        v = phi[:, col]; fv = Xs.iloc[:, col].values.astype(float)
        lo, hi = np.nanpercentile(fv, [5, 95])
        cvals = np.clip((fv - lo) / max(hi - lo, 1e-9), 0, 1)
        jit = rng.normal(0, 0.11, len(v))
        s = ax[1].scatter(v, j + jit, c=cvals, cmap="coolwarm", s=7, alpha=0.65, linewidths=0)
    ax[1].set_yticks(range(len(show))); ax[1].set_yticklabels(show)
    ax[1].axvline(0, color="#555555", lw=1.0)
    ax[1].set_xlabel("Shapley value: effect on the predicted change in AQI")
    ax[1].set_title("(b) How feature values move the forecast")
    cb = fig.colorbar(s, ax=ax[1], pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.set_label("Feature value")
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_shap.png", dpi=DPI)
    plt.close(fig)


def fig5_waterfall():
    phi = np.load(f"{TAB}/d_shap_values.npy")
    idx = np.load(f"{TAB}/d_shap_index.npy")
    base = np.load(f"{TAB}/d_shap_base.npy")
    Xs = pd.read_pickle(f"{TAB}/d_shap_X.pkl")
    P = pd.read_csv(f"{RES}/predictions_h1.csv.gz", parse_dates=["TargetDate"])
    sub = P.iloc[idx].reset_index(drop=True)
    cand = sub[(sub.City == "Kanpur")]
    if len(cand) == 0:
        cand = sub
    j = int(cand.y.idxmax())
    from stage_d import PRETTY
    pretty = np.array([PRETTY.get(f, f) for f in Xs.columns])
    v = phi[j]
    o = np.argsort(-np.abs(v))[:10]
    o = o[np.argsort(v[o])]
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    cols = [RED if x > 0 else BLUE for x in v[o]]
    ax.barh(np.arange(len(o)), v[o], color=cols, height=0.7)
    ax.set_yticks(np.arange(len(o)))
    ax.set_yticklabels([f"{pretty[k]}  =  {Xs.iloc[j, k]:.0f}" for k in o])
    ax.axvline(0, color="#555555", lw=1.0)
    ax.set_xlabel("Shapley value: effect on the predicted change in AQI")
    row = sub.iloc[j]
    ax.set_title(f"{row.City}, target {row.TargetDate.date()}:  observed AQI {row.y:.0f}, "
                 f"forecast {np.clip(row['RASE (proposed)'],0,500):.0f}\n"
                 f"baseline change {base[j]:+.1f} + contributions = predicted change")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_waterfall.png", dpi=DPI)
    plt.close(fig)
    return dict(city=str(row.City), date=str(row.TargetDate.date()), y=float(row.y),
                pred=float(np.clip(row["RASE (proposed)"], 0, 500)), base=float(base[j]))


if __name__ == "__main__":
    audit = json.load(open(f"{OUT}/audit.json"))
    meta = json.load(open(f"{RES}/run_h1.json"))
    Ps = {h: pd.read_csv(f"{RES}/predictions_h{h}.csv.gz", parse_dates=["Date", "TargetDate"])
          for h in (1, 2, 3)}
    T = pd.read_csv(f"{TAB}/b1_main_multihorizon.csv", index_col=[0, 1])
    T1 = T.loc[1]
    fig1_workflow(audit, meta); print("fig1")
    fig2_models(T1, Ps); print("fig2")
    if os.path.exists(f"{RES}/c6_conformal.csv"):
        fig3_city_and_coverage(Ps[1], pd.read_csv(f"{RES}/c6_conformal.csv")); print("fig3")
    if os.path.exists(f"{TAB}/d_shap_values.npy"):
        fig4_shap(); print("fig4")
        info = fig5_waterfall(); print("fig5", info)
        json.dump(info, open(f"{TAB}/d_waterfall.json", "w"), indent=1)
