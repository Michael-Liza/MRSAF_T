# MRSAF — Método de Regresión Supervisada para el Ahorro Financiero

Tesis II · UNMSM-FISI · T7 · Liza Romaldo, Michael Alexander · Asesor: Mg. Daniel A. Quinto Pazce

Pipeline reproducible (F1–F7) que predice el **ahorro objetivo mensual** (`Desired_Savings`) con Random Forest, lo explica con SHAP y lo convierte en una **recomendación según el ahorro predicho** (segmento Bajo / Medio / Alto).

## Estructura
| Ruta | Contenido |
|---|---|
| `mrsaf_pipeline.py` | F1–F6: preprocesamiento, EDA, variables derivadas, entrenamiento (Lineal, Random Forest, XGBoost), evaluación 80/20, SHAP y reglas |
| `mrsaf_demo.py` | Demo: perfil → ahorro predicho → segmento → recomendación + 3 factores SHAP |
| `notebooks/MRSAF_T7.ipynb` | El mismo pipeline dividido por fases |
| `data/data.csv` | Dataset público de finanzas personales (20 000 × 27) |
| `results.json`, `figures/` | Métricas y figuras generadas por el pipeline |
| `docs/` | Diagramas (PNG 220 ppp) y guion del video |

## Ejecución
```bash
pip install -r requirements.txt
python mrsaf_pipeline.py      # ~15-20 min en 2 núcleos (GridSearchCV + SHAP); genera results.json, figures/, models/rf.joblib
python mrsaf_demo.py          # 3 perfiles de ejemplo del dataset
python mrsaf_demo.py --custom # perfil escrito a mano
```
Semilla fija (`random_state = 42`); partición 80 % entrenamiento / 20 % prueba. El modelo entrenado pesa ~280 MB y no se versiona: se regenera con el pipeline.

## Decisiones de diseño clave
- **Objetivo:** `Desired_Savings` (ahorro objetivo), no el ahorro realizado.
- **Control anti-fuga:** se excluyen `Disposable_Income` y `Potential_Savings_*` (derivadas del ingreso, el gasto o el objetivo). El escenario principal excluye además `Desired_Savings_Percentage`.
- **Variables derivadas** (solo con ingreso y gasto): `total_exp`, `disc_exp`, `ess_exp`, `exp_ratio`, `disc_ratio`, `debt_ratio`, `rent_ratio`.
- **Reglas por ahorro predicho:** Bajo (< cuartil 1 del ahorro de entrenamiento) → reducir gasto discrecional; Medio → optimizar deuda y gastos fijos; Alto (> cuartil 3) → fondo de emergencia e inversión. Umbrales configurables.

## Resultados (20 % de prueba, escenario principal)
| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| Regresión lineal | 0.932 | 2 224 | 1 362 |
| Random Forest (GridSearchCV) | 0.919 | 2 412 | 608 |
| XGBoost (GridSearchCV) | 0.867 | 3 096 | 648 |

OOB R² del RF = 0.969; validación cruzada de 5 pliegues = 0.958 ± 0.021; exactitud de segmento = 0.899. El modelo lineal obtiene un R² algo mayor; el Random Forest reduce el MAE a menos de la mitad y se explica de forma nativa con SHAP.

## Limitaciones (declaradas)
- El dataset es público y de aspecto sintético: valida el método, **no** representa a jóvenes universitarios de Lima. La encuesta propia queda como trabajo futuro.
- La importancia SHAP está dominada por `Income`.
- El CTR ≥ 40 % del dashboard aún no se mide (requiere usuarios; semanas 12–13).
- Verificar la licencia del dataset antes de publicar el repositorio.
