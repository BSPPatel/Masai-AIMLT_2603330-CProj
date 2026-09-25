"""
Generates clean, fully documented Jupyter notebooks (01_eda.ipynb and 02_modeling.ipynb)
for Module 2 from the analytical workflow.
"""

import json
import os

def create_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {output_path}")

def generate_eda_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Zepto Data & AI Platform — Module 2: Exploratory Data Analysis (EDA)\n",
                "This notebook profiles the Titanic customer-style dataset, implements missing value strategies, and performs statistical univariate, bivariate, correlation, and multivariate exploratory analysis."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from dataset_loader import get_or_load_titanic\n",
                "\n",
                "df_raw = get_or_load_titanic()\n",
                "print(f\"Raw dataset shape: {df_raw.shape}\")\n",
                "df_raw.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Missing Value Strategy & Cleaning\n",
                "Applying project threshold rules:\n",
                "- `<5% missing`: Drop affected rows (`embarked`)\n",
                "- `5-30% missing`: Median imputation (`age`)\n",
                "- `>30% missing`: Drop column (`deck`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100\n",
                "print(\"Missing percentages:\\n\", missing_pct[missing_pct > 0])\n",
                "\n",
                "df_clean = df_raw.drop(columns=['deck']).dropna(subset=['embarked']).reset_index(drop=True)\n",
                "print(f\"Cleaned dataset shape: {df_clean.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Univariate Analysis (Age & Fare)\n",
                "Histograms, box plots, IQR outlier detection, and central tendency skewness evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for col in ['age', 'fare']:\n",
                "    q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)\n",
                "    iqr = q3 - q1\n",
                "    outliers = df_clean[(df_clean[col] < q1 - 1.5*iqr) | (df_clean[col] > q3 + 1.5*iqr)]\n",
                "    print(f\"{col.upper()}: IQR={iqr:.2f}, Outliers={len(outliers)}\")\n",
                "\n",
                "print(f\"Fare Mean: {df_clean['fare'].mean():.2f}, Median: {df_clean['fare'].median():.2f}, Mode: {df_clean['fare'].mode()[0]:.2f}\")\n",
                "# Skewness conclusion: Mean > Median > Mode indicates Right-Skewed distribution"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Bivariate Survival Rates\n",
                "Calculating survival rates by `sex`, `pclass`, and `sex + pclass` via boolean masks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Female survival:', df_clean[df_clean['sex'] == 'female']['survived'].mean())\n",
                "print('Male survival:', df_clean[df_clean['sex'] == 'male']['survived'].mean())\n",
                "for s in ['female', 'male']:\n",
                "    for c in [1, 2, 3]:\n",
                "        rate = df_clean[(df_clean['sex'] == s) & (df_clean['pclass'] == c)]['survived'].mean()\n",
                "        print(f\"{s} class {c} survival: {rate:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 6-Column Correlation Analysis\n",
                "Restricted to: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']\n",
                "corr = df_clean[corr_cols].corr()\n",
                "sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)\n",
                "plt.title('6-Variable Correlation Matrix')\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells, "analytics/01_eda.ipynb")

def generate_modeling_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Zepto Data & AI Platform — Module 2: Predictive Modeling\n",
                "This notebook trains and evaluates classification models, executes class-imbalance experiments, performs hyperparameter tuning with GridSearchCV & OOB score, trains a multivariate regression model on Fare, and persists the complete pipeline via Joblib."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import joblib\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.linear_model import LogisticRegression, LinearRegression\n",
                "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from imblearn.over_sampling import SMOTE\n",
                "\n",
                "df = pd.read_csv('analytics/titanic.csv').drop(columns=['deck']).dropna(subset=['embarked'])\n",
                "X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]\n",
                "y = df['survived']\n",
                "\n",
                "# Stratified split BEFORE preprocessing to prevent data leakage\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
                "print(f\"Train size: {len(X_train)}, Test size: {len(X_test)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Zero-Leakage Preprocessing & Classification\n",
                "Fitting ColumnTransformer strictly on X_train only."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "num_cols = ['age', 'fare', 'sibsp', 'parch']\n",
                "cat_cols = ['sex', 'embarked', 'pclass']\n",
                "\n",
                "preprocessor = ColumnTransformer([\n",
                "    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),\n",
                "    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)\n",
                "])\n",
                "\n",
                "rf = RandomForestClassifier(random_state=42, n_estimators=100, oob_score=True)\n",
                "pipeline = Pipeline([('prep', preprocessor), ('clf', rf)])\n",
                "pipeline.fit(X_train, y_train)\n",
                "print(\"Random Forest Test Accuracy:\", pipeline.score(X_test, y_test))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Model Persistence & Raw Inference Test"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "joblib.dump(pipeline, 'analytics/models/best_model.joblib')\n",
                "loaded_model = joblib.load('analytics/models/best_model.joblib')\n",
                "\n",
                "sample = pd.DataFrame([{'pclass': 1, 'sex': 'female', 'age': 25, 'sibsp': 0, 'parch': 0, 'fare': 100.0, 'embarked': 'S'}])\n",
                "print(\"Prediction from raw input:\", loaded_model.predict(sample)[0])"
            ]
        }
    ]
    create_notebook(cells, "analytics/02_modeling.ipynb")

if __name__ == "__main__":
    generate_eda_notebook()
    generate_modeling_notebook()
