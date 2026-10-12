"""MRSAF - Pipeline reproducible (F1-F6). Tesis II, T7 - Liza Romaldo.
Uso: python mrsaf_pipeline.py   -> genera results.json, figures/*.png, models/rf.joblib
"""
import time, json, warnings, numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
from xgboost import XGBRegressor
warnings.filterwarnings("ignore"); SEED = 42; TARGET = "Desired_Savings"
EXP = ['Rent','Loan_Repayment','Insurance','Groceries','Transport','Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous']
DISC = ['Eating_Out','Entertainment','Miscellaneous']; ESS = ['Rent','Insurance','Groceries','Transport','Utilities','Healthcare']
LEAK = ['Disposable_Income'] + [f'Potential_Savings_{c}' for c in ['Groceries','Transport','Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous']]
R = {}
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

# ---------- F1 Preprocesamiento ----------
df = pd.read_csv("data/data.csv")
R["n_rows"], R["n_cols"] = df.shape; R["nulls"] = int(df.isna().sum().sum()); R["duplicates"] = int(df.duplicated().sum())
R["neg_values_expenses"] = int((df[EXP] < 0).sum().sum())
# ---------- F2 EDA ----------
num = df.select_dtypes("number")
sp = num.corr(method="spearman")[TARGET].drop(TARGET).sort_values(ascending=False)
R["spearman_target"] = sp.round(3).to_dict(); R["skew"] = num.skew().round(2).to_dict()
R["target_desc"] = df[TARGET].describe().round(1).to_dict(); R["target_zero_pct"] = round(float((df[TARGET] == 0).mean() * 100), 2)
q1, q3 = df.Income.quantile([.25, .75]); R["income_outliers_iqr_pct"] = round(float(((df.Income > q3 + 1.5*(q3-q1))).mean()*100), 2)
# ---------- F3 Feature engineering (sin usar el objetivo) ----------
df["total_exp"] = df[EXP].sum(axis=1); df["disc_exp"] = df[DISC].sum(axis=1); df["ess_exp"] = df[ESS].sum(axis=1)
df["exp_ratio"] = df.total_exp/df.Income; df["disc_ratio"] = df.disc_exp/df.Income
df["debt_ratio"] = df.Loan_Repayment/df.Income; df["rent_ratio"] = df.Rent/df.Income
ENG = ['total_exp','disc_exp','ess_exp','exp_ratio','disc_ratio','debt_ratio','rent_ratio']
BASE = ['Income','Age','Dependents','Occupation','City_Tier'] + EXP
def build(extra=()):
    return pd.get_dummies(df[BASE + ENG + list(extra)], dtype=float)
y = df[TARGET]
scen = {"E1_principal": build(), "E2_con_pct": build(["Desired_Savings_Percentage"])}
R["leak_excluded"] = LEAK
# ---------- F4/F5 Modelado y evaluación (80/20) ----------
def m(yt, p): return dict(R2=r2_score(yt,p), RMSE=mean_squared_error(yt,p)**.5, MAE=mean_absolute_error(yt,p))
R["models"] = {}
for sn, X in scen.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED)
    R["split"] = dict(train=len(Xtr), test=len(Xte), n_features=X.shape[1])
    out = {}
    log("modelos "+sn)
    out["Lineal"] = m(yte, LinearRegression().fit(Xtr,ytr).predict(Xte))
    rf0 = RandomForestRegressor(150, random_state=SEED, oob_score=True, n_jobs=-1).fit(Xtr,ytr)
    out["RF_base"] = {**m(yte, rf0.predict(Xte)), "OOB_R2": rf0.oob_score_}
    xg0 = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=.05, random_state=SEED, n_jobs=-1).fit(Xtr,ytr)
    out["XGB_base"] = m(yte, xg0.predict(Xte))
    if sn == "E1_principal":
        gs = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
              {"n_estimators":[200], "max_depth":[None,16], "min_samples_leaf":[1,3], "max_features":[0.6,1.0]},
              cv=3, scoring="r2", n_jobs=-1).fit(Xtr,ytr)
        log("gs rf listo"); rf = gs.best_estimator_; R["gridsearch"] = dict(best_params=gs.best_params_, cv_r2=gs.best_score_)
        rf_oob = RandomForestRegressor(**{**rf.get_params(), "oob_score":True}).fit(Xtr,ytr)
        out["RF_GridSearch"] = {**m(yte, rf.predict(Xte)), "OOB_R2": rf_oob.oob_score_}
        gx = GridSearchCV(XGBRegressor(random_state=SEED, n_jobs=-1),
              {"n_estimators":[300,600], "max_depth":[3,5], "learning_rate":[.05]}, cv=5, scoring="r2", n_jobs=-1).fit(Xtr,ytr)
        out["XGB_GridSearch"] = m(yte, gx.best_estimator_.predict(Xte)); R["gridsearch_xgb"] = gx.best_params_
        log("gs xgb listo"); cv = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
        R["rf_cv5"] = dict(mean=cv.mean(), std=cv.std())
        # log-target (robustez ante cola de ingresos)
        rl = RandomForestRegressor(**rf.get_params()).fit(Xtr, np.log1p(ytr))
        out["RF_logTarget"] = m(yte, np.expm1(rl.predict(Xte)))
        best_rf, E = rf, (Xtr, Xte, ytr, yte)
    R["models"][sn] = out
# ---------- F6 SHAP + reglas por ahorro predicho ----------
Xtr, Xte, ytr, yte = E; p = best_rf.predict(Xte)
log("shap"); sv = shap.TreeExplainer(best_rf).shap_values(Xte.sample(600, random_state=SEED)); imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values(ascending=False)
R["shap_top"] = imp.head(10).round(1).to_dict()
T1, T2 = float(ytr.quantile(.25)), float(ytr.quantile(.75)); R["thresholds"] = dict(bajo_lt=round(T1,1), alto_gt=round(T2,1))
def seg(v): return np.where(v < T1, "Bajo", np.where(v <= T2, "Medio", "Alto"))
log("reglas"); sr, sp_ = seg(yte.values), seg(p); labs = ["Bajo","Medio","Alto"]
R["segment_accuracy"] = accuracy_score(sr, sp_); R["segment_confusion"] = confusion_matrix(sr, sp_, labels=labs).tolist()
RULES = {"Bajo":"Reducir gasto discrecional (comer fuera, entretenimiento, varios)",
         "Medio":"Optimizar deuda y gastos fijos (préstamos, servicios, renta)",
         "Alto":"Fortalecer fondo de emergencia e invertir el excedente"}
def recommend(row: pd.DataFrame, top_k=3):
    Xr = pd.get_dummies(row, dtype=float).reindex(columns=Xte.columns, fill_value=0.0)
    pr = float(best_rf.predict(Xr)[0]); s = str(seg(np.array([pr]))[0])
    sv1 = shap.TreeExplainer(best_rf).shap_values(Xr)[0]
    drv = pd.Series(sv1, index=Xr.columns).abs().sort_values(ascending=False).head(top_k).index.tolist()
    return dict(ahorro_predicho=round(pr,1), segmento=s, recomendacion=RULES[s], factores_shap=drv)
joblib.dump(dict(model=best_rf, columns=list(Xte.columns), T1=T1, T2=T2), "models/rf.joblib")
demo = df.sample(3, random_state=7)
R["demo"] = []
for i, r_ in demo.iterrows():
    row = build().loc[[i]]
    # reconstruye fila cruda con categorias
    raw = df.loc[[i], BASE + ENG]; rec = recommend(raw); rec["real"] = round(float(df.loc[i, TARGET]),1); R["demo"].append(rec)
log("figuras")
# ---------- Figuras ----------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = "#1F4E79"
fig, ax = plt.subplots(1,2, figsize=(9,3.6)); ax[0].hist(df[TARGET], bins=60, color=C); ax[0].set_title("Desired_Savings"); ax[0].set_xlabel("Ahorro deseado"); ax[0].set_ylabel("Frecuencia")
ax[1].hist(np.log1p(df.Income), bins=60, color="#7A9CC6"); ax[1].set_title("log(1+Income)"); ax[1].set_xlabel("log(1+Ingreso)"); plt.tight_layout(); plt.savefig("figures/fig_dist.png", dpi=200); plt.close()
s = sp.drop([c for c in sp.index if c in ("Disposable_Income",) or c.startswith("Potential")]).sort_values()
fig, ax = plt.subplots(figsize=(6.5,5)); ax.barh(s.index, s.values, color=C); ax.set_xlabel("Correlación de Spearman con Desired_Savings"); plt.tight_layout(); plt.savefig("figures/fig_spearman.png", dpi=200); plt.close()
mm = R["models"]["E1_principal"]; names = ["Lineal","RF_GridSearch","XGB_GridSearch"]
fig, ax = plt.subplots(figsize=(5.5,3.4)); ax.bar(["Lineal","Random Forest","XGBoost"], [mm[n]["R2"] for n in names], color=[ "#7A9CC6", C, "#A9A9A9"]); ax.axhline(.75, ls="--", c="r", lw=1); ax.set_ylabel("R² (test, 20 %)"); ax.set_ylim(0,1); ax.text(2.45,.77,"meta 0.75",color="r",ha="right",fontsize=9); plt.tight_layout(); plt.savefig("figures/fig_modelos.png", dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(4.8,4.4)); ax.scatter(yte, p, s=4, alpha=.35, color=C); lim=[0, max(yte.max(), p.max())]; ax.plot(lim, lim, "r--", lw=1); ax.set_xlabel("Ahorro real"); ax.set_ylabel("Ahorro predicho"); plt.tight_layout(); plt.savefig("figures/fig_pred_real.png", dpi=200); plt.close()
t = imp.head(10)[::-1]; fig, ax = plt.subplots(figsize=(6.2,3.8)); ax.barh(t.index, t.values, color=C); ax.set_xlabel("Media |valor SHAP|"); plt.tight_layout(); plt.savefig("figures/fig_shap.png", dpi=200); plt.close()
cm = np.array(R["segment_confusion"]); fig, ax = plt.subplots(figsize=(4.2,3.6)); ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3): ax.text(j,i,cm[i,j],ha="center",va="center",color="w" if cm[i,j]>cm.max()/2 else "k")
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(labs); ax.set_xlabel("Segmento predicho"); ax.set_ylabel("Segmento real"); plt.tight_layout(); plt.savefig("figures/fig_segmentos.png", dpi=200); plt.close()
json.dump(R, open("results.json","w"), indent=2, default=float)
print(json.dumps(R, indent=1, default=float))
