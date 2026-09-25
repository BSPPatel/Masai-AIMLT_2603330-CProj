# Module 2: Analytics & Machine Learning Pipeline

## Overview
This module implements an end-to-end analytical and machine learning pipeline on the Titanic dataset, following strict zero-data-leakage principles, statistical exploratory analysis, hyperparameter tuning, class-imbalance experiments, regression diagnostics, and complete pipeline persistence.

$$\textbf{LOAD} \longrightarrow \textbf{PROFILE} \longrightarrow \textbf{CLEAN} \longrightarrow \textbf{EXPLORE} \longrightarrow \textbf{PREPROCESS} \longrightarrow \textbf{MODEL} \longrightarrow \textbf{TUNE} \longrightarrow \textbf{EVALUATE} \longrightarrow \textbf{PERSIST}$$

---

## 1. Dataset & Profiling (`titanic.csv`)
- **Source:** Seaborn standard Titanic dataset (`sns.load_dataset("titanic")`), fetched and committed to `analytics/titanic.csv` for guaranteed 100% offline reproducibility.
- **Initial Profile:** 891 rows, 15 columns.
- **Missing Value Strategy (Project Threshold Rule):**
  - **$<5\%$ missing:** Dropped 2 affected rows in `embarked` / `embark_town` (0.22% missing).
  - **$5\%\text{--}30\%$ missing:** Handled via median imputation for `age` (19.87% missing) strictly inside the training pipeline.
  - **$>30\%$ missing:** Dropped `deck` column entirely (77.10% missing) to avoid high-dimensional sparsity and noise.
  - **Resulting Clean Dataset:** 889 rows, 14 columns.

---

## 2. Statistical Exploratory Data Analysis (EDA)

### Univariate Analysis & Skewness
- **`age`:** $Q_1 = 20.12$, $Q_3 = 38.00$, $\text{IQR} = 17.88$. Outliers detected: 7 (0.98%).
- **`fare`:** $Q_1 = 7.90$, $Q_3 = 31.00$, $\text{IQR} = 23.10$. Outliers detected: 114 (12.82%).
- **Fare Central Tendency & Skewness:**
  $$\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$$
  *Conclusion:* The `fare` distribution exhibits heavy positive (**right-skewed**) asymmetry driven by luxury first-class suites.

### Bivariate Survival Rates (Boolean Masking)
- **By Sex:** Female: $74.04\%$ vs. Male: $18.89\%$
- **By Class:** 1st Class: $62.62\%$ | 2nd Class: $47.28\%$ | 3rd Class: $24.24\%$
- **By Sex + Class Interaction:**
  - 1st Class Females: $96.74\%$ | 2nd Class Females: $92.11\%$ | 3rd Class Females: $50.00\%$
  - 1st Class Males: $36.89\%$ | 2nd Class Males: $15.74\%$ | 3rd Class Males: $13.54\%$

### 6-Variable Correlation Analysis
Constrained strictly to `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (excluding derived flags `adult_male` and `alone`):
1. **`pclass` $\leftrightarrow$ `fare` ($-0.548$):** Strongest negative correlation reflecting ticket pricing tiers where upper-deck cabins commanded exponentially higher ticket prices.
2. **`sibsp` $\leftrightarrow$ `parch` ($+0.415$):** Strongest positive correlation demonstrating family co-travel patterns (siblings/spouses traveling with parents/children).

### Multivariate Data Story (Visual Artifacts in `charts/`)
1. **`multivariate_1_survival_sex_class.png`:** Visualizes how socio-economic privilege compounded gender protocol. Over 96% of upper-class women survived, whereas 3rd class men experienced near-total casualty rates (86.5% perish rate).
2. **`multivariate_2_age_class_survival.png`:** Highlights age distribution across survival tiers. Younger passengers in 2nd and 3rd class had significantly higher survival odds, reflecting priority child evacuation.
3. **`multivariate_3_fare_age_survival.png`:** Shows fare clustering against age. The vast cluster of sub-£20 tickets contains almost all casualties across working-age adults.
4. **`multivariate_4_family_survival.png`:** Demonstrates an inverted-U survival curve with family size. Solo travelers had ~30% survival; small families (2–4 members) peaked at 55–70% survival; large families (5+) suffered due to evacuation coordination breakdowns.

---

## 3. Machine Learning & Zero-Leakage Preprocessing

### Stratified Split & Leakage Prevention
- **Split:** $80\%$ Train (711 rows) / $20\%$ Test (178 rows), stratified on `survived` to preserve the identical 61.8% / 38.2% class distribution.
- **Leakage Prevention:** Preprocessing `ColumnTransformer` is fit **strictly on `X_train`** and applied downstream to `X_test`.

### Side-by-Side Classification Model Comparison
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8146 | 0.7966 | 0.6912 | 0.7402 | 0.8596 | `[[98, 12], [21, 47]]` |
| **Decision Tree** | 0.8034 | 0.8000 | 0.6471 | 0.7154 | 0.8481 | `[[99, 11], [24, 44]]` |
| **Random Forest** | 0.8146 | 0.7692 | 0.7353 | 0.7519 | 0.8151 | `[[95, 15], [18, 50]]` |

*Decision Tree architecture diagram saved to `charts/decision_tree.png` and ROC curves saved to `charts/roc_curves.png`.*

### Class-Imbalance Experiment
| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Unweighted)** | 0.7966 | 0.6912 | 0.7402 |
| **`class_weight='balanced'`** | 0.7246 | 0.7353 | 0.7299 |
| **SMOTE (Train Fold Only)** | 0.7353 | 0.7353 | 0.7353 |

*Imbalance Finding:* Applying class balancing boosted minority class recall from 0.6912 to 0.7353, ensuring critical rescue prioritization.

### Hyperparameter Tuning & Out-Of-Bag (OOB) Score
- **`GridSearchCV` Best Parameters:** `{'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 100}`
- **Best Cross-Validation F1 Score:** $0.7618$
- **Random Forest OOB Score:** $0.8158$ (confirming strong generalization without test data leakage).

---

## 4. Regression Side Task: Predicting Fare
- **Model:** Multivariate Linear Regression using `pclass`, `sex`, `age`, `sibsp`, `parch`, and `embarked`.
- **Metrics:**
  - **MAE:** $17.85$
  - **RMSE:** $40.55$
  - **$R^2$:** $0.3838$
  - **Adjusted $R^2$:** $0.3621$
- **Heteroscedasticity Analysis:** Residual plot (`charts/residual_plot.png`) displays a clear funnel shape where error variance expands dramatically for high-fare tickets, proving pronounced heteroscedasticity.

---

## 5. Deployment Recommendation & Persistence
- **Deployment Recommendation:** Random Forest is recommended for production. It demonstrates the highest balanced test performance (F1: 0.752, OOB: 0.816) while naturally modeling non-linear socio-demographic interactions without vulnerability to multicollinearity.
- **Persisted Artifact:** The complete pipeline (imputer + onehot encoder + standard scaler + tuned Random Forest) is serialized to:
  `analytics/models/best_model.joblib`
- **Raw Inference Verification:** Reloading the joblib model and passing unscaled raw JSON features directly produces verified predictions:
  ```python
  loaded_pipe = joblib.load("analytics/models/best_model.joblib")
  loaded_pipe.predict(pd.DataFrame([{"pclass": 1, "sex": "female", "age": 28.0, "sibsp": 0, "parch": 0, "fare": 85.5, "embarked": "S"}]))
  # Returns: array([1]) -> Survived
  ```

---

## 6. Execution Instructions
Run the entire analytical pipeline:
```bash
.venv\Scripts\python.exe analytics/eda_and_modeling.py
```
Run automated test suite:
```bash
.venv\Scripts\pytest.exe tests/test_analytics.py -v
```
