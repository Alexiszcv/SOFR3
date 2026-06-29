"""
SOFR–IORB spread regression on Treasury flows.
y_t = Δ(SOFR − IORB)_t ~ tga_drain + tga_drain×reserves_centered
                         + gross_settlement + rrp + dummies
FOMC days excluded; Newey-West (HAC) standard errors.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA = "/Users/alexis/Downloads/Summer 2026 - BNP/Projets/SOFR/Data"

# ── FOMC decision dates 2021-2026 (day IORB / fed funds target changes) ──────
FOMC_DATES = pd.to_datetime([
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
])

# Tax-deadline months/days (approximate; refined via tax_receipts if available)
# Mid-April (Apr-15), mid-June (Jun-15), mid-Sep (Sep-15), mid-Jan (Jan-15)
# Quarterly corp estimates: Apr-15, Jun-15, Sep-15, Dec-15
TAX_MONTHS_DAYS = [(1, 15), (3, 15), (4, 15), (6, 15), (9, 15), (12, 15)]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_fred(path, col):
    """Generic FRED CSV → daily Series indexed by date."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("observation_date")[col].rename(col)


def load_sofr():
    return load_fred(f"{DATA}/SOFR.csv", "SOFR")


def load_iorb():
    return load_fred(f"{DATA}/IORB.csv", "IORB")


def load_rrp():
    """RRP take-up in $bn (raw: millions)."""
    s = load_fred(f"{DATA}/RRPONTSYD.csv", "RRPONTSYD")
    return (s / 1_000).rename("rrp")


def load_reserves():
    """WRESBAL weekly → forward-filled daily series, in $bn."""
    s = load_fred(f"{DATA}/WRESBAL.csv", "WRESBAL")
    return (s / 1_000).rename("reserves")


def load_tga_close():
    """
    DTS Operating Cash Balance → TGA closing balance in $bn (daily).
    Column 'Opening Balance Today' holds the balance for every row type
    (misleading label; it is the balance at the given date for that account type).
    """
    df = pd.read_csv(
        f"{DATA}/DTS_OpCashBal_20210624_20260623.csv",
        encoding="utf-8-sig",   # strips BOM
        parse_dates=["Record Date"],
        low_memory=False,
    )
    close = df[df["Type of Account"].str.contains("Closing Balance", na=False)].copy()
    close["value"] = pd.to_numeric(close["Opening Balance Today"], errors="coerce")
    close = (
        close.groupby("Record Date")["value"]
        .sum()
        .sort_index()
        / 1_000           # → $bn
    )
    close.index.name = "date"
    return close.rename("tga_close")


def load_gross_settlement():
    """
    DTS Public Debt Transactions → gross_settlement = issues + redemptions
    for Marketable Bills / Notes / Bonds only, in $bn.
    """
    df = pd.read_csv(
        f"{DATA}/DTS_PubDebtTrans_20210624_20260623.csv",
        parse_dates=["Record Date"],
        low_memory=False,
    )
    mask = (
        (df["Security Marketability"] == "Marketable")
        & (df["Security Type"].isin(["Bills", "Notes", "Bonds"]))
    )
    df = df[mask].copy()
    df["Transactions Today"] = pd.to_numeric(df["Transactions Today"], errors="coerce")
    pivot = (
        df.groupby(["Record Date", "Transaction Type"])["Transactions Today"]
        .sum()
        .unstack(fill_value=0)
        / 1_000   # → $bn
    )
    pivot.index.name = "date"
    pivot.columns.name = None

    # Ensure both columns exist even if one is missing in data
    for col in ["Issues", "Redemptions"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot["gross_settlement"] = pivot["Issues"] + pivot["Redemptions"]
    pivot["net_issuance"] = pivot["Issues"] - pivot["Redemptions"]   # descriptive only
    return pivot[["gross_settlement", "net_issuance"]]


def load_tax_receipts():
    """
    DTS Deposits/Withdrawals → tax_receipts in $bn.
    Aggregates inflow categories whose names match known tax labels.
    """
    df = pd.read_csv(
        f"{DATA}/DTS_OpCashDpstWdrl_20210624_20260623.csv",
        parse_dates=["Record Date"],
        low_memory=False,
    )
    tax_keywords = [
        "taxes", "tax", "ftd", "income", "withheld",
        "fica", "seca", "individual income",
    ]
    pattern = "|".join(tax_keywords)
    mask = (
        (df["Transaction Type"] == "Deposits")
        & (df["Transaction Category"].str.lower().str.contains(pattern, na=False))
    )
    df = df[mask].copy()
    df["Transactions Today"] = pd.to_numeric(df["Transactions Today"], errors="coerce")
    out = (
        df.groupby("Record Date")["Transactions Today"]
        .sum()
        / 1_000
    ).rename("tax_receipts")
    out.index.name = "date"
    return out


def load_dealer_take():
    """
    treasury.csv (semicolon-delimited) → dealer_take in $bn per IssueDate.
    PrimaryDealerAccepted summed by settlement date.
    """
    df = pd.read_csv(f"{DATA}/treasury.csv", sep=";", low_memory=False)
    df["IssueDate"] = pd.to_datetime(df["IssueDate"], errors="coerce")
    df["PrimaryDealerAccepted"] = pd.to_numeric(
        df["PrimaryDealerAccepted"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    out = (
        df.groupby("IssueDate")["PrimaryDealerAccepted"]
        .sum()
        / 1e9           # $ → $bn
    ).rename("dealer_take")
    out.index.name = "date"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build panel
# ─────────────────────────────────────────────────────────────────────────────

def build_panel():
    print("Loading data…")
    sofr       = load_sofr()
    iorb       = load_iorb()
    rrp        = load_rrp()
    reserves   = load_reserves()
    tga_close  = load_tga_close()
    dts_debt   = load_gross_settlement()
    tax_rec    = load_tax_receipts()
    dealer     = load_dealer_take()

    # Spine = SOFR trading dates (Fed repo market open days)
    spine = sofr.index.sort_values()

    panel = pd.DataFrame(index=spine)
    panel.index.name = "date"
    panel["SOFR"]   = sofr.reindex(spine)
    panel["IORB"]   = iorb.reindex(spine)
    panel["rrp"]    = rrp.reindex(spine)
    panel["tga_close"] = tga_close.reindex(spine)

    # Weekly reserves → forward-fill on daily spine
    panel["reserves"] = reserves.reindex(spine).ffill()

    # DTS daily flows (business days already)
    panel = panel.join(dts_debt.reindex(spine), how="left")
    panel["tax_receipts"] = tax_rec.reindex(spine)
    panel["dealer_take"]  = dealer.reindex(spine).fillna(0.0)

    # ── Target: Δ(SOFR − IORB) ───────────────────────────────────────────────
    panel["spread"] = panel["SOFR"] - panel["IORB"]
    panel["y"]      = panel["spread"].diff()

    # ── TGA drain: + = reserves drained ──────────────────────────────────────
    # close.diff() > 0  →  TGA balance rose  →  banks paid cash into TGA
    # (via debt issuance / tax receipts)  →  bank reserves fell = drain.
    panel["tga_drain"] = panel["tga_close"].diff()

    # ── Dummies ───────────────────────────────────────────────────────────────
    panel["fomc"]    = panel.index.isin(FOMC_DATES).astype(int)
    panel["eom"]     = (
        (panel.index + pd.offsets.BDay(1)).month != panel.index.month
    ).astype(int)
    panel["eoq"]     = (
        panel["eom"] & panel.index.month.isin([3, 6, 9, 12])
    ).astype(int)
    panel["tax_day"] = panel.index.map(
        lambda d: int(any(
            d.month == m and abs(d.day - day) <= 2
            for m, day in TAX_MONTHS_DAYS
        ))
    )

    # ── Reserve regime: centred (mean = 0 over full sample) ──────────────────
    panel["reserves_centered"] = panel["reserves"] - panel["reserves"].mean()
    panel["interaction"]       = panel["tga_drain"] * panel["reserves_centered"]

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Regression
# ─────────────────────────────────────────────────────────────────────────────

REGRESSORS = [
    "tga_drain",
    "interaction",
    "gross_settlement",
    "rrp",
    "eom",
    "eoq",
    "tax_day",
    "dealer_take",
]


def estimate(panel, include_dealer=True):
    cols = ["y"] + REGRESSORS
    reg_cols = REGRESSORS if include_dealer else [c for c in REGRESSORS if c != "dealer_take"]

    # Drop FOMC days and rows with any NaN in the variables of interest
    sample = panel[panel["fomc"] == 0][["y"] + reg_cols].dropna()

    print(f"\nEstimation sample: {len(sample)} obs  "
          f"({sample.index[0].date()} → {sample.index[-1].date()})")

    y = sample["y"]
    X = sm.add_constant(sample[reg_cols])

    # Newey-West (HAC) – bandwidth ~ 4*sqrt(T/100) ≈ floor rule
    nw_lags = int(np.floor(4 * (len(sample) / 100) ** (2 / 9)))
    print(f"Newey-West lags: {nw_lags}")

    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return res, sample


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def check_sign_priors(res):
    priors = {
        "tga_drain":       (">", 0),
        "interaction":     ("<", 0),
        "gross_settlement":(">", 0),
        "rrp":             ("<", 0),
    }
    print("\n── Sign priors ──────────────────────────────")
    for var, (sign, val) in priors.items():
        if var not in res.params.index:
            continue
        coef = res.params[var]
        pval = res.pvalues[var]
        ok   = (coef > val) if sign == ">" else (coef < val)
        flag = "✓" if ok else "✗"
        print(f"  {flag}  {var:25s}  coef={coef:+.4f}  p={pval:.3f}  prior β{sign}0")


def compute_vif(sample, reg_cols):
    X = sm.add_constant(sample[reg_cols].dropna())
    vif = pd.DataFrame({
        "feature": X.columns,
        "VIF":     [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).set_index("feature")
    return vif[vif.index != "const"]


def run_diagnostics(res, sample, reg_cols):
    print("\n── OLS summary (HAC SE) ─────────────────────")
    print(res.summary())

    check_sign_priors(res)

    vif = compute_vif(sample, reg_cols)
    print("\n── VIF ──────────────────────────────────────")
    print(vif.to_string())

    resid = res.resid
    dw    = durbin_watson(resid)
    lb    = acorr_ljungbox(resid, lags=[10], return_df=True)
    print(f"\n── Serial correlation ───────────────────────")
    print(f"  Durbin-Watson: {dw:.3f}  (2=no autocorr)")
    print(f"  Ljung-Box Q(10): stat={lb['lb_stat'].iloc[0]:.2f}  "
          f"p={lb['lb_pvalue'].iloc[0]:.3f}")

    return resid


def oos_rmse(panel, reg_cols, split_date="2024-01-01"):
    """Chronological train/test split; fit on train, score on test."""
    full = panel[panel["fomc"] == 0][["y"] + reg_cols].dropna()
    train = full[full.index < split_date]
    test  = full[full.index >= split_date]
    if len(test) < 20:
        print("\nTest set too small for OOS evaluation.")
        return None

    nw_lags = int(np.floor(4 * (len(train) / 100) ** (2 / 9)))
    res_tr  = sm.OLS(train["y"], sm.add_constant(train[reg_cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    X_te    = sm.add_constant(test[reg_cols], has_constant="add")
    yhat    = res_tr.predict(X_te)
    rmse    = np.sqrt(np.mean((test["y"] - yhat) ** 2))
    bench   = np.sqrt(np.mean((test["y"] - test["y"].mean()) ** 2))   # mean-forecast bench
    print(f"\n── Out-of-sample (test from {split_date}) ──")
    print(f"  Test obs:   {len(test)}")
    print(f"  OOS RMSE:   {rmse:.5f} bp/day")
    print(f"  Bench RMSE: {bench:.5f} bp/day  (forecast = train mean)")
    print(f"  Skill:      {1 - rmse/bench:.3f}  (>0 beats mean forecast)")
    return rmse, bench, yhat, test["y"]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_panel(panel):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    panel["spread"].plot(ax=axes[0], color="steelblue", lw=1)
    axes[0].set_ylabel("SOFR−IORB (bp)")
    axes[0].set_title("SOFR–IORB Spread")

    panel["tga_close"].plot(ax=axes[1], color="darkorange", lw=1)
    axes[1].set_ylabel("$bn")
    axes[1].set_title("TGA Closing Balance")

    panel["rrp"].plot(ax=axes[2], color="seagreen", lw=1)
    axes[2].set_ylabel("$bn")
    axes[2].set_title("Overnight RRP Take-up")

    panel["reserves"].plot(ax=axes[3], color="purple", lw=1)
    axes[3].set_ylabel("$bn")
    axes[3].set_title("Bank Reserves (WRESBAL, fwd-filled)")

    for ax in axes:
        for d in FOMC_DATES:
            ax.axvline(d, color="red", alpha=0.15, lw=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SOFR–IORB Spread and Treasury Flow Variables", fontsize=13)
    plt.tight_layout()
    plt.savefig("/Users/alexis/Downloads/Summer 2026 - BNP/Projets/SOFR/panel_overview.png",
                dpi=150, bbox_inches="tight")
    print("\nSaved: panel_overview.png")


def plot_residuals(res, sample, oos_result=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Fitted vs actual
    ax = axes[0, 0]
    ax.scatter(res.fittedvalues, sample["y"], alpha=0.3, s=10, color="steelblue")
    mn = min(res.fittedvalues.min(), sample["y"].min())
    mx = max(res.fittedvalues.max(), sample["y"].max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Actual Δ(SOFR−IORB)")
    ax.set_title("Fitted vs Actual")
    ax.grid(True, alpha=0.3)

    # Residuals over time
    ax = axes[0, 1]
    res.resid.plot(ax=ax, color="steelblue", lw=0.7, alpha=0.8)
    ax.axhline(0, color="red", lw=1)
    ax.set_title("Residuals over Time")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[1, 0]
    res.resid.hist(bins=60, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.grid(True, alpha=0.3)

    # OOS predicted vs actual (if available)
    ax = axes[1, 1]
    if oos_result is not None:
        _, _, yhat, y_test = oos_result
        ax.plot(y_test.index, y_test.values, label="Actual", lw=1, color="steelblue")
        ax.plot(yhat.index, yhat.values, label="OOS Forecast", lw=1,
                color="darkorange", linestyle="--")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("Out-of-Sample Forecast")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle("Regression Diagnostics", fontsize=13)
    plt.tight_layout()
    plt.savefig("/Users/alexis/Downloads/Summer 2026 - BNP/Projets/SOFR/diagnostics.png",
                dpi=150, bbox_inches="tight")
    print("Saved: diagnostics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Robustness
# ─────────────────────────────────────────────────────────────────────────────

def robustness_checks(panel):
    print("\n\n══════════════════════════════════════════════")
    print("  ROBUSTNESS CHECKS")
    print("══════════════════════════════════════════════")

    base_cols = ["tga_drain", "interaction", "gross_settlement", "rrp",
                 "eom", "eoq", "tax_day"]

    # A) Regime split instead of interaction
    print("\n── A) Regime split (high vs low reserves) ───")
    med = panel["reserves"].median()
    panel["hi_reserves"] = (panel["reserves"] >= med).astype(int)
    panel["tga_drain_hi"] = panel["tga_drain"] * panel["hi_reserves"]
    panel["tga_drain_lo"] = panel["tga_drain"] * (1 - panel["hi_reserves"])
    regime_cols = ["tga_drain_hi", "tga_drain_lo", "gross_settlement", "rrp",
                   "eom", "eoq", "tax_day"]
    sample_r = panel[panel["fomc"] == 0][["y"] + regime_cols].dropna()
    nw = int(np.floor(4 * (len(sample_r) / 100) ** (2 / 9)))
    res_r = sm.OLS(sample_r["y"],
                   sm.add_constant(sample_r[regime_cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw}
    )
    print(f"  R² = {res_r.rsquared:.4f}  (vs base interaction model)")
    print(res_r.params[["tga_drain_hi", "tga_drain_lo"]].to_string())

    # B) With FOMC days included
    print("\n── B) Including FOMC days ────────────────────")
    sample_b = panel[["y"] + base_cols].dropna()
    nw = int(np.floor(4 * (len(sample_b) / 100) ** (2 / 9)))
    res_b = sm.OLS(sample_b["y"],
                   sm.add_constant(sample_b[base_cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw}
    )
    print(f"  R²={res_b.rsquared:.4f}  (FOMC excluded: see main model)")

    # C) net_issuance instead of tga_drain
    print("\n── C) net_issuance as primary regresssor ─────")
    alt_cols = ["net_issuance", "gross_settlement", "rrp", "eom", "eoq", "tax_day"]
    sample_c = panel[panel["fomc"] == 0][["y"] + alt_cols].dropna()
    nw = int(np.floor(4 * (len(sample_c) / 100) ** (2 / 9)))
    res_c = sm.OLS(sample_c["y"],
                   sm.add_constant(sample_c[alt_cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw}
    )
    print(f"  R²={res_c.rsquared:.4f}  net_issuance coef={res_c.params.get('net_issuance', float('nan')):.5f}")

    # D) Without dummies
    print("\n── D) Without dummies ────────────────────────")
    nodummy_cols = ["tga_drain", "interaction", "gross_settlement", "rrp"]
    sample_d = panel[panel["fomc"] == 0][["y"] + nodummy_cols].dropna()
    nw = int(np.floor(4 * (len(sample_d) / 100) ** (2 / 9)))
    res_d = sm.OLS(sample_d["y"],
                   sm.add_constant(sample_d[nodummy_cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw}
    )
    print(f"  R²={res_d.rsquared:.4f}  (with dummies: see main model)")

    print("\nDone with robustness checks.")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panel = build_panel()

    print(f"\nPanel shape: {panel.shape}")
    print(f"Date range:  {panel.index[0].date()} → {panel.index[-1].date()}")
    print("\nMissing values:")
    print(panel[["y", "tga_drain", "interaction", "gross_settlement",
                 "rrp", "reserves"]].isna().sum())

    # Save panel
    out_path = "/Users/alexis/Downloads/Summer 2026 - BNP/Projets/SOFR/panel_daily.csv"
    panel.to_csv(out_path)
    print(f"\nPanel saved → {out_path}")

    # Main regression (with dealer_take)
    reg_cols = ["tga_drain", "interaction", "gross_settlement", "rrp",
                "eom", "eoq", "tax_day", "dealer_take"]
    res, sample = estimate(panel, include_dealer=True)

    # Diagnostics
    run_diagnostics(res, sample, reg_cols)

    # OOS
    oos_result = oos_rmse(panel, reg_cols, split_date="2024-01-01")

    # Plots
    plot_panel(panel)
    plot_residuals(res, sample, oos_result)

    # Robustness
    robustness_checks(panel)
