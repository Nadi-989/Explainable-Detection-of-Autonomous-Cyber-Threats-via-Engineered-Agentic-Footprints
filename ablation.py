"""
============================================================================
 ablation.py — Ablation study of the feature families (Section VI-C, Table VIII)
============================================================================
Measures how much each feature family contributes to detection performance by
comparing three configurations under stratified 5-fold cross-validation, on the
best model per dataset:

    (A) Raw only          : original dataset features only
    (B) + Engineered      : raw + standard engineered features (Eqs. 1-4)
    (C) + Agentic (full)  : raw + engineered + agentic footprint (Eqs. 5-10)

For each configuration it reports F1, MCC (primary), and ROC-AUC, and prints the
MCC gain attributable to the agentic features. Results are written to
ablation_results.csv.

Key finding (see the paper): on these balanced benchmarks the agentic features
change accuracy only marginally; their value is primarily INTERPRETIVE (they
rank first on spam under SHAP), not accuracy-driven. The script is written to
make that honest comparison easy to reproduce.

USAGE
    python ablation.py

The path resolver matches main.py: it reads the three CSVs from ./data if that
folder exists, otherwise from the repository root (flat layout). Requires:
numpy, pandas, scikit-learn.
============================================================================
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef

# ---- paths (portable: ./data if present, else repo root) -----------------
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
DATA = os.path.join(HERE, "data") if os.path.isdir(os.path.join(HERE, "data")) else HERE

EPS = 1e-6
MAX_ROWS = 10000          # subsample cap for runtime (matches main.py)
K = 5                     # cross-validation folds

NSL_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


# ==========================================================================
# Each *_sets() returns (raw, engineered, full, y, best_model)
#   raw        : original features only
#   engineered : raw + standard engineered features
#   full       : engineered + agentic footprint
# ==========================================================================
def nsl_sets():
    df = pd.read_csv(os.path.join(DATA, "nsl_train.csv"), header=None, names=NSL_COLS)
    y = (df["label"] != "normal").astype(int)
    raw = pd.get_dummies(df.drop(columns=["label"]),
                         columns=["protocol_type", "service", "flag"],
                         drop_first=True).astype(float)
    # (B) standard engineered
    eng = raw.copy()
    eng["bytes_ratio"]    = eng["src_bytes"] / (eng["dst_bytes"] + EPS)
    eng["total_bytes"]    = eng["src_bytes"] + eng["dst_bytes"]
    eng["log_src_bytes"]  = np.log1p(eng["src_bytes"].clip(lower=0))
    eng["log_dst_bytes"]  = np.log1p(eng["dst_bytes"].clip(lower=0))
    eng["error_rate_sum"] = eng["serror_rate"] + eng["rerror_rate"]
    eng["srv_diversity"]  = eng["srv_count"] / (eng["count"] + EPS)
    eng["high_error_flag"] = (eng["error_rate_sum"] > 0.5).astype(float)
    # (C) + agentic footprint
    full = eng.copy()
    full["ag_superhuman_rate"] = full["count"] / (full["duration"] + 1.0)
    full["ag_priv_escalation"] = full["root_shell"] + full["su_attempted"] + (full["num_root"] > 0).astype(float)
    full["ag_persistence"]     = full["num_file_creations"] + full["num_shells"] + full["num_compromised"]
    full["ag_lateral_scan"]    = full["dst_host_count"] * full["diff_srv_rate"]
    full["ag_no_human_pause"]  = ((full["duration"] < 1) & (full["count"] > 20)).astype(float)
    full["ag_autonomy_score"]  = ((full["ag_superhuman_rate"] > full["ag_superhuman_rate"].quantile(0.9)).astype(float)
                                  + full["ag_priv_escalation"] + (full["ag_persistence"] > 0).astype(float)
                                  + full["ag_no_human_pause"])
    model = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42)
    return "NSL-KDD", raw, eng, full, y, model


def phishing_sets():
    df = pd.read_csv(os.path.join(DATA, "phishing.csv")).drop(columns=["Domain"], errors="ignore")
    y = df["Label"].astype(int)
    raw = df.drop(columns=["Label"]).astype(float)
    eng = raw.copy()
    eng["depth_per_length"] = eng["URL_Depth"] / (eng["URL_Length"] + 1)
    risk = [c for c in ["Have_IP","Have_At","Redirection","Prefix/Suffix","TinyURL",
                        "iFrame","Mouse_Over","Right_Click","Web_Forwards"] if c in eng]
    eng["risk_flag_sum"] = eng[risk].sum(axis=1)
    eng["secure_established"] = eng["https_Domain"] * eng["Domain_Age"]
    full = eng.copy()
    young = (full["Domain_Age"] == 0).astype(float)
    struct = sum(full[c] for c in ["Prefix/Suffix","TinyURL","Redirection","Have_At"] if c in full)
    full["ag_mass_generated"]      = young + struct
    full["ag_page_automation"]     = sum(full[c] for c in ["iFrame","Web_Forwards","Redirection"] if c in full)
    full["ag_no_user_interaction"] = sum(full[c] for c in ["Right_Click","Mouse_Over"] if c in full)
    full["ag_autonomy_score"]      = full["ag_mass_generated"] + full["ag_page_automation"] + full["ag_no_user_interaction"]
    model = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42)
    return "Phishing", raw, eng, full, y, model


def spam_sets():
    df = pd.read_csv(os.path.join(DATA, "spambase.csv"))
    df.columns = [c.strip().strip('"') for c in df.columns]
    tgt = df.columns[-1]
    y = df[tgt].astype(int)
    raw = df.drop(columns=[tgt]).astype(float)
    eng = raw.copy()
    wc = [c for c in eng.columns if c.startswith("word_freq")]
    cc = [c for c in eng.columns if c.startswith("char_freq")]
    cap = [c for c in eng.columns if "capital" in c.lower()]
    eng["word_freq_total"] = eng[wc].sum(axis=1)
    eng["char_freq_total"] = eng[cc].sum(axis=1)
    if len(cap) >= 2:
        eng["cap_run_ratio"] = eng[cap[0]] / (eng[cap[-1]] + EPS)
    sp = [c for c in ["word_freq_free","word_freq_money","word_freq_credit",
                     "word_freq_business","word_freq_000"] if c in eng]
    eng["spammy_words"] = eng[sp].sum(axis=1)
    full = eng.copy()
    a, l, t = "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"
    if l in full: full["ag_capital_burst"]  = full[l] / (full[a] + EPS)
    if t in full: full["ag_capital_volume"] = np.log1p(full[t].clip(lower=0))
    templ = [c for c in ["word_freq_free","word_freq_money","word_freq_credit",
                        "word_freq_business","word_freq_000","word_freq_order",
                        "word_freq_receive"] if c in full]
    full["ag_template_density"] = full[templ].sum(axis=1)
    urg = [c for c in ["char_freq_%21","char_freq_%24"] if c in full]
    if urg:
        full["ag_urgency_chars"] = full[urg].sum(axis=1)
    comps = [c for c in ["ag_capital_burst","ag_template_density","ag_urgency_chars"] if c in full]
    z = full[comps].apply(lambda s: (s - s.mean()) / (s.std() + EPS))
    full["ag_autonomy_score"] = z.sum(axis=1)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
    return "Spambase", raw, eng, full, y, model


# ==========================================================================
# Evaluation
# ==========================================================================
CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
SCORING = {"f1": "f1", "mcc": make_scorer(matthews_corrcoef), "roc_auc": "roc_auc"}

def score(X, y, model):
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=0)
        y = y.loc[X.index]
    r = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    return r["test_f1"].mean(), r["test_mcc"].mean(), r["test_roc_auc"].mean()


def main():
    print("=" * 64)
    print(" Ablation study — contribution of each feature family (5-fold)")
    print("=" * 64)
    print(f"{'Dataset':10s} {'Config':16s} {'#feat':>6s} {'F1':>7s} {'MCC':>7s} {'AUC':>7s}")
    print("-" * 64)

    rows = []
    for loader in (nsl_sets, phishing_sets, spam_sets):
        name, raw, eng, full, y, model = loader()
        mcc_by_cfg = {}
        for cfg, X in [("Raw only", raw), ("+ Engineered", eng), ("+ Agentic (full)", full)]:
            f1, mcc, auc = score(X, y, model)
            mcc_by_cfg[cfg] = mcc
            rows.append(dict(dataset=name, config=cfg, n_features=X.shape[1],
                             f1=f1, mcc=mcc, auc=auc))
            print(f"{name:10s} {cfg:16s} {X.shape[1]:>6d} {f1:>7.3f} {mcc:>7.3f} {auc:>7.3f}")
        gain = mcc_by_cfg["+ Agentic (full)"] - mcc_by_cfg["+ Engineered"]
        print(f"{'':10s} >>> MCC gain from agentic features: {gain:+.4f}\n")

    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(HERE, "ablation_results.csv"), index=False)
    print("Saved ablation_results.csv")
    print("\nNote: a small accuracy gain is expected on these balanced benchmarks;")
    print("the agentic features' primary value is interpretive (top SHAP rank on spam),")
    print("not accuracy-driven. See Section VI-C of the paper.")


if __name__ == "__main__":
    main()
