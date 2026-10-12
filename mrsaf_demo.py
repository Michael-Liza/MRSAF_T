"""Demo MRSAF: de un perfil financiero a un ahorro predicho y una recomendación.
Requiere haber ejecutado mrsaf_pipeline.py (genera models/rf.joblib).
Uso:  python mrsaf_demo.py            -> 3 perfiles de ejemplo del dataset
      python mrsaf_demo.py --custom   -> un perfil escrito a mano
"""
import sys, time, warnings, joblib, shap, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
EXP = ['Rent','Loan_Repayment','Insurance','Groceries','Transport','Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous']
RULES = {"Bajo": "Reducir gasto discrecional (comer fuera, entretenimiento, varios)",
         "Medio": "Optimizar deuda y gastos fijos (préstamos, servicios, renta)",
         "Alto": "Fortalecer fondo de emergencia e invertir el excedente"}
art = joblib.load("models/rf.joblib"); RF, COLS, T1, T2 = art["model"], art["columns"], art["T1"], art["T2"]
EXPL = shap.TreeExplainer(RF)

def features(p: dict) -> pd.DataFrame:
    d = pd.DataFrame([p]); I = d.Income
    d["total_exp"] = d[EXP].sum(axis=1); d["disc_exp"] = d[['Eating_Out','Entertainment','Miscellaneous']].sum(axis=1)
    d["ess_exp"] = d[['Rent','Insurance','Groceries','Transport','Utilities','Healthcare']].sum(axis=1)
    d["exp_ratio"] = d.total_exp / I; d["disc_ratio"] = d.disc_exp / I; d["debt_ratio"] = d.Loan_Repayment / I; d["rent_ratio"] = d.Rent / I
    X = pd.get_dummies(d[['Income','Age','Dependents','Occupation','City_Tier'] + EXP + ['total_exp','disc_exp','ess_exp','exp_ratio','disc_ratio','debt_ratio','rent_ratio']], dtype=float)
    return X.reindex(columns=COLS, fill_value=0.0)

def recomendar(p: dict):
    t = time.time(); X = features(p)
    yhat = max(0.0, float(RF.predict(X)[0]))                                   # ahorro predicho
    seg = "Bajo" if yhat < T1 else ("Medio" if yhat <= T2 else "Alto")         # regla por ahorro predicho
    phi = pd.Series(EXPL.shap_values(X)[0], index=COLS)
    top = phi.abs().sort_values(ascending=False).head(3).index.tolist()
    return dict(ahorro_predicho=round(yhat, 1), segmento=seg, recomendacion=RULES[seg], factores_shap=top, segundos=round(time.time()-t, 2))

if __name__ == "__main__":
    if "--custom" in sys.argv:
        perfil = dict(Income=30000, Age=24, Dependents=0, Occupation="Student", City_Tier="Tier_1", Rent=6000, Loan_Repayment=0, Insurance=600,
                      Groceries=3500, Transport=1500, Eating_Out=1200, Entertainment=900, Utilities=1500, Healthcare=500, Education=2500, Miscellaneous=400)
        print(f"Umbrales: Bajo < {T1:.1f} <= Medio <= {T2:.1f} < Alto"); print(recomendar(perfil))
    else:
        df = pd.read_csv("data/data.csv"); print(f"Umbrales: Bajo < {T1:.1f} <= Medio <= {T2:.1f} < Alto\n")
        for i, r in df.sample(3, random_state=7).iterrows():
            out = recomendar(r.to_dict()); out["ahorro_objetivo_real"] = round(float(r.Desired_Savings), 1); print(out)
