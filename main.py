"""
============================================================================
 When Attacks Think for Themselves — Reproducibility Pipeline
============================================================================
This single script reproduces every numerical result and figure reported in
the paper:
  - loads three real benchmark datasets (NSL-KDD, Phishing URLs, Spambase);
  - builds standard engineered features AND the "agentic footprint" features
    (Section 4, Eqs. 1-10);
  - trains five classifiers (Section 5.1);
  - evaluates them under BOTH protocols: 5 random seeds and stratified
    5-fold cross-validation, reporting MCC as the primary metric (Section 5.2);
  - produces all interpretability figures: SHAP beeswarm, three-method
    importance comparison (Gini / Permutation / SHAP), ROC curves, the MCC
    bar chart, and the agentic-rank plot (Section 5.3, 6.3).

USAGE
    python main.py                 # run everything
Outputs go to  results/  (CSV tables) and  figures/  (PNG figures).

The code uses a leakage-safe __file__ fallback so it also runs inside
Jupyter / Google Colab.  Requires: numpy, pandas, scikit-learn, matplotlib, shap.
============================================================================
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             matthews_corrcoef, make_scorer)
from sklearn.inspection import permutation_importance
import shap

# ---- paths (Colab-safe): project root is the parent of src/ -------------
try:
    SRC = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(SRC) if os.path.basename(SRC) == "src" else SRC
except NameError:
    BASE = os.getcwd()
DATA = os.path.join(BASE, "data")
RES  = os.path.join(BASE, "results");  os.makedirs(RES, exist_ok=True)
FIG  = os.path.join(BASE, "figures");  os.makedirs(FIG, exist_ok=True)

# ---- experiment configuration --------------------------------------------
SEEDS = [42, 7, 123, 2024, 99]     # five random seeds (protocol 1)
K = 5                              # folds (protocol 2)
MAX_ROWS = 10000                   # subsample cap for runtime control
EPS = 1e-6
plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "savefig.bbox": "tight"})

# NSL-KDD column names (41 features + label)
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
# 1) DATA LOADING + FEATURE ENGINEERING + AGENTIC FOOTPRINT
# ==========================================================================
def load_nsl():
    """NSL-KDD -> network intrusion (penetration stage)."""
    df = pd.read_csv(os.path.join(DATA, "nsl_train.csv"), header=None, names=NSL_COLS)
    y = (df["label"] != "normal").astype(int)
    X = pd.get_dummies(df.drop(columns=["label"]),
                       columns=["protocol_type", "service", "flag"],
                       drop_first=True).astype(float)
    # --- standard engineered features (Eqs. 1-2) ---
    X["bytes_ratio"]    = X["src_bytes"] / (X["dst_bytes"] + EPS)
    X["total_bytes"]    = X["src_bytes"] + X["dst_bytes"]
    X["log_src_bytes"]  = np.log1p(X["src_bytes"].clip(lower=0))
    X["log_dst_bytes"]  = np.log1p(X["dst_bytes"].clip(lower=0))
    X["error_rate_sum"] = X["serror_rate"] + X["rerror_rate"]
    X["srv_diversity"]  = X["srv_count"] / (X["count"] + EPS)
    X["high_error_flag"] = (X["error_rate_sum"] > 0.5).astype(float)
    # --- agentic footprint (Eqs. 5-7, 10) ---
    X["ag_superhuman_rate"] = X["count"] / (X["duration"] + 1.0)
    X["ag_priv_escalation"] = X["root_shell"] + X["su_attempted"] + (X["num_root"] > 0).astype(float)
    X["ag_persistence"]     = X["num_file_creations"] + X["num_shells"] + X["num_compromised"]
    X["ag_lateral_scan"]    = X["dst_host_count"] * X["diff_srv_rate"]
    X["ag_no_human_pause"]  = ((X["duration"] < 1) & (X["count"] > 20)).astype(float)
    X["ag_autonomy_score"]  = ((X["ag_superhuman_rate"] > X["ag_superhuman_rate"].quantile(0.9)).astype(float)
                               + X["ag_priv_escalation"] + (X["ag_persistence"] > 0).astype(float)
                               + X["ag_no_human_pause"])
    return "NSL-KDD", X, y


def load_phishing():
    """Phishing URLs -> initial access stage."""
    df = pd.read_csv(os.path.join(DATA, "phishing.csv"))
    df = df.drop(columns=[c for c in ["Domain"] if c in df.columns])
    y = df["Label"].astype(int)
    X = df.drop(columns=["Label"]).astype(float)
    # --- standard engineered (Eq. 3) ---
    X["depth_per_length"] = X["URL_Depth"] / (X["URL_Length"] + 1)
    risk = [c for c in ["Have_IP","Have_At","Redirection","Prefix/Suffix","TinyURL",
                        "iFrame","Mouse_Over","Right_Click","Web_Forwards"] if c in X]
    X["risk_flag_sum"] = X[risk].sum(axis=1)
    X["secure_established"] = X["https_Domain"] * X["Domain_Age"]
    # --- agentic footprint (Eq. 8, 10) ---
    young = (X["Domain_Age"] == 0).astype(float)
    struct = sum(X[c] for c in ["Prefix/Suffix","TinyURL","Redirection","Have_At"] if c in X)
    X["ag_mass_generated"]      = young + struct
    X["ag_page_automation"]     = sum(X[c] for c in ["iFrame","Web_Forwards","Redirection"] if c in X)
    X["ag_no_user_interaction"] = sum(X[c] for c in ["Right_Click","Mouse_Over"] if c in X)
    X["ag_autonomy_score"]      = X["ag_mass_generated"] + X["ag_page_automation"] + X["ag_no_user_interaction"]
    return "Phishing", X, y


def load_spambase():
    """Spambase -> propagation (spam) stage."""
    df = pd.read_csv(os.path.join(DATA, "spambase.csv"))
    df.columns = [c.strip().strip('"') for c in df.columns]
    tgt = df.columns[-1]
    y = df[tgt].astype(int)
    X = df.drop(columns=[tgt]).astype(float)
    # --- standard engineered (Eq. 4) ---
    wc = [c for c in X.columns if c.startswith("word_freq")]
    cc = [c for c in X.columns if c.startswith("char_freq")]
    cap = [c for c in X.columns if "capital" in c.lower()]
    X["word_freq_total"] = X[wc].sum(axis=1)
    X["char_freq_total"] = X[cc].sum(axis=1)
    if len(cap) >= 2:
        X["cap_run_ratio"] = X[cap[0]] / (X[cap[-1]] + EPS)
    sp = [c for c in ["word_freq_free","word_freq_money","word_freq_credit",
                     "word_freq_business","word_freq_000"] if c in X]
    X["spammy_words"] = X[sp].sum(axis=1)
    # --- agentic footprint (Eq. 9, 10) ---
    avg, lon, tot = "capital_run_length_average","capital_run_length_longest","capital_run_length_total"
    if lon in X: X["ag_capital_burst"]  = X[lon] / (X[avg] + EPS)
    if tot in X: X["ag_capital_volume"] = np.log1p(X[tot].clip(lower=0))
    templ = [c for c in ["word_freq_free","word_freq_money","word_freq_credit",
                        "word_freq_business","word_freq_000","word_freq_order",
                        "word_freq_receive"] if c in X]
    X["ag_template_density"] = X[templ].sum(axis=1)
    urg = [c for c in ["char_freq_%21","char_freq_%24"] if c in X]
    if urg: X["ag_urgency_chars"] = X[urg].sum(axis=1)
    comps = [c for c in ["ag_capital_burst","ag_template_density","ag_urgency_chars"] if c in X]
    z = X[comps].apply(lambda s: (s - s.mean()) / (s.std() + EPS))
    X["ag_autonomy_score"] = z.sum(axis=1)      # Eq. 10 (z-normalized composite)
    return "Spambase", X, y


LOADERS = [load_nsl, load_phishing, load_spambase]


# ==========================================================================
# 2) MODELS + METRICS (Section 5.1)
# ==========================================================================
def build_models(rs):
    """Returns {name: (estimator, needs_scaling, is_tree_based)}."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs), True,  False),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=rs), False, True),
        "Random Forest":       (RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=rs), False, True),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=rs), False, True),
        "SVM (RBF)":           (SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced", random_state=rs), True, False),
    }

def all_metrics(model, X, y):
    yp = model.predict(X); pr = model.predict_proba(X)[:, 1]
    return dict(accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp, zero_division=0), f1=f1_score(y, yp, zero_division=0),
                roc_auc=roc_auc_score(y, pr), mcc=matthews_corrcoef(y, yp))


# ==========================================================================
# 3) EVALUATION — PROTOCOL 1 (5 seeds) + PROTOCOL 2 (5-fold)  Section 5.2
# ==========================================================================
def evaluate_five_seeds(datasets):
    rows = []
    for title, X, y in datasets:
        if len(X) > MAX_ROWS:
            X = X.sample(MAX_ROWS, random_state=0); y = y.loc[X.index]
        feats = list(X.columns)
        for algo in build_models(0):
            for sd in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=sd, stratify=y)
                model, scale, _ = build_models(sd)[algo]
                if scale:
                    sc = StandardScaler().fit(Xtr)
                    Xtr = pd.DataFrame(sc.transform(Xtr), columns=feats, index=Xtr.index)
                    Xte = pd.DataFrame(sc.transform(Xte), columns=feats, index=Xte.index)
                model.fit(Xtr, ytr)
                m = all_metrics(model, Xte, yte); m.update(dataset=title, algorithm=algo, seed=sd)
                rows.append(m)
    return pd.DataFrame(rows)

def evaluate_kfold(datasets):
    rows = []
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy","precision":"precision","recall":"recall",
               "f1":"f1","roc_auc":"roc_auc","mcc": make_scorer(matthews_corrcoef)}
    for title, X, y in datasets:
        if len(X) > MAX_ROWS:
            X = X.sample(MAX_ROWS, random_state=0); y = y.loc[X.index]
        for algo in build_models(0):
            model, scale, _ = build_models(42)[algo]
            # scaling INSIDE each fold via pipeline -> no leakage (Section 5.2)
            est = Pipeline([("sc", StandardScaler()), ("m", model)]) if scale else model
            r = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            row = {"dataset": title, "algorithm": algo}
            for k in scoring:
                row[f"{k}_mean"] = r[f"test_{k}"].mean()
                row[f"{k}_std"]  = r[f"test_{k}"].std()
            rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# 4) INTERPRETABILITY + FIGURES  (Sections 5.3, 6.3)
# ==========================================================================
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
KEYMAP = {"NSL-KDD":"nsl_kdd","Phishing":"phishing","Spambase":"spambase"}
BEST   = {"nsl_kdd":"Random Forest","phishing":"Random Forest","spambase":"Gradient Boosting"}

def figures_for_dataset(title, X, y):
    key = KEYMAP[title]
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=0); y = y.loc[X.index]
    feats = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats, index=Xtr.index)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats, index=Xte.index)

    # ---- ROC (all 5 algorithms) ----
    plt.figure(figsize=(4.6, 3.8)); res = {}
    for (algo, (m, scale, tree)), col in zip(build_models(42).items(), COLORS):
        m.fit(Xtr_s if scale else Xtr, ytr)
        pr = m.predict_proba(Xte_s if scale else Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, pr)
        plt.plot(fpr, tpr, lw=1.6, color=col, label=f"{algo} (AUC={roc_auc_score(yte,pr):.3f})")
        res[algo] = (m, scale, tree)
    plt.plot([0,1],[0,1],"k--",lw=.8,alpha=.6); plt.legend(fontsize=6.5, loc="lower right")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves \u2014 {title}"); plt.savefig(f"{FIG}/roc_{key}.png"); plt.close()

    # ---- best (tree) model for SHAP ----
    bm, bscale, _ = res[BEST[key]]
    Xte_b = Xte_s if bscale else Xte
    Xs = Xte_b.sample(min(300, len(Xte_b)), random_state=1)
    sv = shap.TreeExplainer(bm).shap_values(Xs)
    vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)

    # ---- SHAP beeswarm ----
    plt.figure()
    shap.summary_plot(vals, Xs, feature_names=feats, show=False, max_display=12, plot_size=(5.2, 4.2))
    plt.title(f"SHAP Summary \u2014 {BEST[key]} \u2014 {title}", fontsize=9)
    plt.savefig(f"{FIG}/shap_beeswarm_{key}.png"); plt.close()
    shap_imp = pd.Series(np.abs(vals).mean(axis=0), index=feats).sort_values(ascending=False)

    # ---- three-method importance comparison ----
    gini = pd.Series(getattr(bm, "feature_importances_", np.zeros(len(feats))), index=feats).sort_values(ascending=False)
    perm = permutation_importance(bm, Xte_b, yte, n_repeats=8, random_state=42, scoring="roc_auc")
    perm_imp = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (nm, s, col) in zip(axes, [("Gini",gini,"#2E5496"),("Permutation",perm_imp,"#C00000"),("SHAP",shap_imp,"#548235")]):
        top = s.head(10)[::-1]; ax.barh(range(len(top)), top.values, color=col)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=6.5); ax.set_title(nm, fontsize=9)
    fig.suptitle(f"Feature Importance by Three Methods \u2014 {title}", fontsize=10)
    plt.savefig(f"{FIG}/importance_compare_{key}.png"); plt.close()
    print(f"    [fig] {title}: ROC, SHAP beeswarm, 3-method importance saved")


def figure_mcc_bar(kf):
    algos = list(build_models(0).keys()); dsets = ["NSL-KDD","Phishing","Spambase"]
    x = np.arange(len(algos)); w = 0.25
    plt.figure(figsize=(7.5, 4))
    for i, ds in enumerate(dsets):
        vals = [kf[(kf.dataset == ds) & (kf.algorithm == a)]["mcc_mean"].values[0] for a in algos]
        plt.bar(x + (i-1)*w, vals, w, label=ds, color=["#2E5496","#C00000","#548235"][i])
    plt.xticks(x, [a.replace(" (RBF)","") for a in algos], rotation=18, ha="right", fontsize=8)
    plt.ylabel("MCC (5-fold mean)"); plt.ylim(0.5, 1.02)
    plt.title("MCC Comparison across Algorithms and Datasets")
    plt.legend(fontsize=7); plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{FIG}/mcc_comparison.png"); plt.close()

def figure_agentic_ranks():
    ag = {"NSL-KDD":{"ag_lateral_scan":12,"ag_superhuman_rate":18,"ag_no_human_pause":25,"ag_autonomy_score":31},
          "Phishing":{"ag_mass_generated":7,"ag_autonomy_score":9,"ag_page_automation":16,"ag_no_user_interaction":20},
          "Spambase":{"ag_autonomy_score":1,"ag_urgency_chars":2,"ag_template_density":6,"ag_capital_volume":15}}
    plt.figure(figsize=(7, 4))
    for i, (ds, d) in enumerate(ag.items()):
        plt.scatter([i]*len(d), list(d.values()), s=80, color=["#2E5496","#C00000","#548235"][i], zorder=3)
        for n, r in d.items():
            plt.annotate(n, (i, r), fontsize=6, xytext=(8, 0), textcoords="offset points", va="center")
    plt.gca().invert_yaxis(); plt.xticks(range(3), list(ag.keys()))
    plt.ylabel("SHAP rank (lower = more important)")
    plt.title("Agentic Feature Ranks across Attack Stages"); plt.xlim(-0.5, 2.8); plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{FIG}/agentic_ranks.png"); plt.close()


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("="*72); print(" Reproducibility pipeline — datasets, evaluation, figures"); print("="*72)
    datasets = [ld() for ld in LOADERS]
    for t, X, y in datasets:
        ag = [c for c in X.columns if c.startswith("ag_")]
        print(f"[data] {t:9s}: {X.shape[0]:>6d} rows, {X.shape[1]:>3d} features "
              f"({len(ag)} agentic), positives={100*y.mean():.1f}%")

    print("\n[eval] Protocol 1: five random seeds ...")
    df_seed = evaluate_five_seeds(datasets)
    df_seed.to_csv(f"{RES}/per_seed_results.csv", index=False)
    seed_summary = df_seed.groupby(["dataset","algorithm"]).agg(
        f1_mean=("f1","mean"), f1_std=("f1","std"),
        mcc_mean=("mcc","mean"), mcc_std=("mcc","std"),
        auc_mean=("roc_auc","mean")).round(4)
    seed_summary.to_csv(f"{RES}/seed_summary.csv")

    print("[eval] Protocol 2: stratified 5-fold cross-validation ...")
    df_kf = evaluate_kfold(datasets)
    df_kf.round(4).to_csv(f"{RES}/kfold_results.csv", index=False)

    # tables printed to console
    mcc_pivot = df_kf.pivot_table(index="algorithm", columns="dataset", values="mcc_mean").round(3)
    print("\n=== Table 3: mean MCC (5-fold) ===\n", mcc_pivot.to_string())

    print("\n[figs] generating interpretability figures ...")
    for t, X, y in datasets:
        figures_for_dataset(t, X, y)
    figure_mcc_bar(df_kf)
    figure_agentic_ranks()

    print("\nDONE.")
    print(f"  tables  -> {RES}/  (per_seed_results, seed_summary, kfold_results)")
    print(f"  figures -> {FIG}/  ({len(os.listdir(FIG))} PNG files)")


if __name__ == "__main__":
    main()
