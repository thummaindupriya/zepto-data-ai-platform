from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# =========================================================
# 1. LOAD DATASET EXACTLY ONCE + OFFLINE FALLBACK
# =========================================================

df = sns.load_dataset("titanic")

# Required offline fallback
df.to_csv(CSV_PATH, index=False)

print("=" * 70)
print("TITANIC DATASET")
print("=" * 70)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# =========================================================
# 2. BASIC INFORMATION + MISSING VALUES
# =========================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
df.info()

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("MISSING VALUE REPORT")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent.round(2)
})

print(missing_report[missing_report["missing_count"] > 0])


# =========================================================
# 3. CLEANING FOR EDA
# =========================================================

eda_df = df.copy()

# Under 5% -> drop affected rows
eda_df = eda_df.dropna(subset=["embarked", "embark_town"])

# 5%-30% -> median imputation
eda_df["age"] = eda_df["age"].fillna(eda_df["age"].median())

# >30% -> drop column
eda_df = eda_df.drop(columns=["deck"])

print("\n" + "=" * 70)
print("CLEANED EDA DATASET")
print("=" * 70)
print("Shape:", eda_df.shape)
print("\nRemaining missing values:")
print(eda_df.isnull().sum()[eda_df.isnull().sum() > 0])


# =========================================================
# 4. UNIVARIATE ANALYSIS
# =========================================================

# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(eda_df["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "age_histogram.png", dpi=150)
plt.close()

# Age boxplot
plt.figure(figsize=(7, 4))
sns.boxplot(x=eda_df["age"])
plt.title("Age Boxplot")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "age_boxplot.png", dpi=150)
plt.close()

# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(eda_df["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fare_histogram.png", dpi=150)
plt.close()

# Fare boxplot
plt.figure(figsize=(7, 4))
sns.boxplot(x=eda_df["fare"])
plt.title("Fare Boxplot")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fare_boxplot.png", dpi=150)
plt.close()


def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((series < lower) | (series > upper)).sum()

    return q1, q3, iqr, lower, upper, count


age_q1, age_q3, age_iqr, age_lower, age_upper, age_outliers = (
    iqr_outlier_count(eda_df["age"])
)

fare_q1, fare_q3, fare_iqr, fare_lower, fare_upper, fare_outliers = (
    iqr_outlier_count(eda_df["fare"])
)

print("\n" + "=" * 70)
print("IQR OUTLIER ANALYSIS")
print("=" * 70)

print(f"Age Q1: {age_q1:.2f}")
print(f"Age Q3: {age_q3:.2f}")
print(f"Age IQR: {age_iqr:.2f}")
print(f"Age lower bound: {age_lower:.2f}")
print(f"Age upper bound: {age_upper:.2f}")
print(f"Age outlier count: {age_outliers}")

print()

print(f"Fare Q1: {fare_q1:.2f}")
print(f"Fare Q3: {fare_q3:.2f}")
print(f"Fare IQR: {fare_iqr:.2f}")
print(f"Fare lower bound: {fare_lower:.2f}")
print(f"Fare upper bound: {fare_upper:.2f}")
print(f"Fare outlier count: {fare_outliers}")


# Fare mean / median / mode
fare_mean = eda_df["fare"].mean()
fare_median = eda_df["fare"].median()
fare_mode = eda_df["fare"].mode().iloc[0]

if fare_mean > fare_median > fare_mode:
    fare_shape = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_shape = "left-skewed"
else:
    fare_shape = "approximately symmetric or mixed"

print("\n" + "=" * 70)
print("FARE CENTRAL TENDENCY")
print("=" * 70)
print(f"Mean: {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode: {fare_mode:.2f}")
print(f"Interpretation: Fare is {fare_shape}.")


# =========================================================
# 5. BIVARIATE SURVIVAL ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("SURVIVAL RATES")
print("=" * 70)

# Boolean masking examples
male_rate = eda_df.loc[eda_df["sex"] == "male", "survived"].mean()
female_rate = eda_df.loc[eda_df["sex"] == "female", "survived"].mean()

first_class_rate = eda_df.loc[eda_df["pclass"] == 1, "survived"].mean()
second_class_rate = eda_df.loc[eda_df["pclass"] == 2, "survived"].mean()
third_class_rate = eda_df.loc[eda_df["pclass"] == 3, "survived"].mean()

print(f"Male survival rate: {male_rate:.3f}")
print(f"Female survival rate: {female_rate:.3f}")

print(f"1st class survival rate: {first_class_rate:.3f}")
print(f"2nd class survival rate: {second_class_rate:.3f}")
print(f"3rd class survival rate: {third_class_rate:.3f}")

sex_pclass_rates = (
    eda_df
    .groupby(["sex", "pclass"], observed=True)["survived"]
    .mean()
    .reset_index(name="survival_rate")
)

print("\nSex + Pclass survival rates:")
print(sex_pclass_rates)


# Combined boolean masking
female_first_class = eda_df.loc[
    (eda_df["sex"] == "female") & (eda_df["pclass"] == 1),
    "survived"
].mean()

male_third_class = eda_df.loc[
    (eda_df["sex"] == "male") & (eda_df["pclass"] == 3),
    "survived"
].mean()

print(f"\nFemale + 1st class survival rate: {female_first_class:.3f}")
print(f"Male + 3rd class survival rate: {male_third_class:.3f}")


# =========================================================
# 6. CORRELATION MATRIX — EXACTLY SIX COLUMNS
# =========================================================

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

corr_matrix = eda_df[corr_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)
print(corr_matrix)

# Find strongest two absolute off-diagonal correlations
pairs = []

for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        col1 = corr_columns[i]
        col2 = corr_columns[j]
        value = corr_matrix.loc[col1, col2]
        pairs.append((col1, col2, value, abs(value)))

pairs.sort(key=lambda x: x[3], reverse=True)

print("\nTop 2 strongest absolute correlations:")
for col1, col2, value, absolute_value in pairs[:2]:
    print(
        f"{col1} vs {col2}: "
        f"correlation={value:.3f}, "
        f"absolute={absolute_value:.3f}"
    )


# =========================================================
# 7. MULTIVARIATE DATA STORY — 4 DISTINCT CHARTS
# =========================================================

# Chart 1: survival by sex and class
plt.figure(figsize=(8, 5))
sns.barplot(
    data=eda_df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Passenger Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "survival_by_sex_class.png", dpi=150)
plt.close()

# Chart 2: age vs fare colored by survival
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=eda_df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)
plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "age_fare_survival.png", dpi=150)
plt.close()

# Chart 3: fare by class and survival
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=eda_df,
    x="pclass",
    y="fare",
    hue="survived"
)
plt.title("Fare Distribution by Class and Survival")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fare_class_survival.png", dpi=150)
plt.close()

# Chart 4: family size relationship
story_df = eda_df.copy()
story_df["family_size"] = story_df["sibsp"] + story_df["parch"] + 1

plt.figure(figsize=(9, 5))
sns.barplot(
    data=story_df,
    x="family_size",
    y="survived"
)
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "survival_family_size.png", dpi=150)
plt.close()


# =========================================================
# 8. EXPLORATORY STANDARDIZATION
# =========================================================

standardized_df = eda_df.copy()

before_age_mean = standardized_df["age"].mean()
before_age_std = standardized_df["age"].std()

before_fare_mean = standardized_df["fare"].mean()
before_fare_std = standardized_df["fare"].std()

standardized_df["age_z"] = (
    standardized_df["age"] - before_age_mean
) / before_age_std

standardized_df["fare_z"] = (
    standardized_df["fare"] - before_fare_mean
) / before_fare_std

print("\n" + "=" * 70)
print("STANDARDIZATION")
print("=" * 70)

print("Before standardization:")
print(
    f"Age mean={before_age_mean:.3f}, "
    f"std={before_age_std:.3f}"
)
print(
    f"Fare mean={before_fare_mean:.3f}, "
    f"std={before_fare_std:.3f}"
)

print("\nAfter standardization:")
print(
    f"Age mean={standardized_df['age_z'].mean():.3f}, "
    f"std={standardized_df['age_z'].std():.3f}"
)
print(
    f"Fare mean={standardized_df['fare_z'].mean():.3f}, "
    f"std={standardized_df['fare_z'].std():.3f}"
)


# =========================================================
# 9. MODELING DATA
#    READS THE SAME SAVED CSV — NO SECOND sns.load_dataset
# =========================================================

model_df = pd.read_csv(CSV_PATH)

# Remove high-missing deck and duplicate target-like descriptive columns
model_df = model_df.drop(
    columns=["deck", "alive", "class", "who", "embark_town"],
    errors="ignore"
)

X = model_df.drop(columns=["survived"])
y = model_df["survived"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("STRATIFIED TRAIN / TEST SPLIT")
print("=" * 70)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Overall survival rate:", y.mean())
print("Training survival rate:", y_train.mean())
print("Testing survival rate:", y_test.mean())


# =========================================================
# 10. PREPROCESSING
# =========================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# =========================================================
# 11. THREE CLASSIFIERS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    ),
}

results = []
fitted_models = {}

for model_name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    })

    fitted_models[model_name] = pipeline

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(
        FIGURE_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150
    )
    plt.close()


results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("CLASSIFIER COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))


# =========================================================
# 12. HYPERPARAMETER TUNING
# =========================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        ),
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("Best Random Forest parameters:")
print(grid_search.best_params_)
print("Best cross-validation F1:", grid_search.best_score_)


# Evaluate tuned Random Forest
tuned_pred = best_rf.predict(X_test)
tuned_prob = best_rf.predict_proba(X_test)[:, 1]

tuned_metrics = {
    "accuracy": accuracy_score(y_test, tuned_pred),
    "precision": precision_score(y_test, tuned_pred, zero_division=0),
    "recall": recall_score(y_test, tuned_pred, zero_division=0),
    "f1": f1_score(y_test, tuned_pred, zero_division=0),
    "auc": roc_auc_score(y_test, tuned_prob),
}

print("\nTuned Random Forest:")
for metric, value in tuned_metrics.items():
    print(f"{metric.capitalize():10s}: {value:.4f}")


# =========================================================
# 13. SELECT BEST CLASSIFIER BY F1
# =========================================================

comparison_for_selection = results_df.copy()

comparison_for_selection.loc[
    len(comparison_for_selection)
] = [
    "Tuned Random Forest",
    tuned_metrics["accuracy"],
    tuned_metrics["precision"],
    tuned_metrics["recall"],
    tuned_metrics["f1"],
    tuned_metrics["auc"],
]

best_classifier_name = comparison_for_selection.loc[
    comparison_for_selection["f1"].idxmax(),
    "model"
]

print("\n" + "=" * 70)
print("BEST CLASSIFIER")
print("=" * 70)
print("Selected using highest test-set F1:")
print(best_classifier_name)


# =========================================================
# 14. SAVE COMPLETE BEST PIPELINE
# =========================================================

if best_classifier_name == "Tuned Random Forest":
    best_pipeline = best_rf
else:
    best_pipeline = fitted_models[best_classifier_name]

MODEL_PATH = BASE_DIR / "best_classifier_pipeline.joblib"

joblib.dump(
    best_pipeline,
    MODEL_PATH
)

print("\nSaved complete preprocessing + model pipeline:")
print(MODEL_PATH)


# =========================================================
# 15. RELOAD AND PREDICT RAW NEW INPUT
# =========================================================

loaded_pipeline = joblib.load(MODEL_PATH)

new_passenger = pd.DataFrame([
    {
        "pclass": 3,
        "sex": "male",
        "age": 30,
        "sibsp": 0,
        "parch": 0,
        "fare": 10.0,
        "embarked": "S",
        "alone": True,
    }
])

new_prediction = loaded_pipeline.predict(new_passenger)[0]
new_probability = loaded_pipeline.predict_proba(
    new_passenger
)[0, 1]

print("\n" + "=" * 70)
print("RELOADED PIPELINE TEST")
print("=" * 70)
print("Prediction:", int(new_prediction))
print("Survival probability:", round(new_probability, 4))


# =========================================================
# 16. FARE REGRESSION
# =========================================================

regression_df = pd.read_csv(CSV_PATH)

regression_df = regression_df.drop(
    columns=[
        "deck",
        "fare",
        "alive",
        "class",
        "who",
        "embark_town",
    ],
    errors="ignore"
)

# Target must come from original dataset
regression_target = pd.read_csv(CSV_PATH)["fare"]

reg_X = regression_df
reg_y = regression_target

reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
    reg_X,
    reg_y,
    test_size=0.20,
    random_state=42
)

reg_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
]

reg_categorical_features = [
    "sex",
    "embarked",
]

reg_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", reg_numeric_transformer, reg_numeric_features),
        ("cat", reg_categorical_transformer, reg_categorical_features),
    ]
)

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression()),
    ]
)

regression_pipeline.fit(reg_X_train, reg_y_train)

reg_pred = regression_pipeline.predict(reg_X_test)

mae = mean_absolute_error(reg_y_test, reg_pred)
rmse = mean_squared_error(
    reg_y_test,
    reg_pred
) ** 0.5
r2 = r2_score(reg_y_test, reg_pred)

n = len(reg_y_test)
p = reg_X_test.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\n" + "=" * 70)
print("FARE REGRESSION")
print("=" * 70)
print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R2         : {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")


# Residual plot
residuals = reg_y_test - reg_pred

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=reg_pred,
    y=residuals,
    alpha=0.7
)
plt.axhline(0, linestyle="--")
plt.title("Fare Regression Residual Plot")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "fare_regression_residuals.png",
    dpi=150
)
plt.close()


# =========================================================
# 17. SAVE RESULT TABLES
# =========================================================

comparison_for_selection.to_csv(
    BASE_DIR / "classifier_comparison.csv",
    index=False
)

pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}]).to_csv(
    BASE_DIR / "regression_metrics.csv",
    index=False
)

missing_report.to_csv(
    BASE_DIR / "missing_value_report.csv"
)

print("\n" + "=" * 70)
print("ANALYTICS COMPLETE")
print("=" * 70)
print("Figures saved to:", FIGURE_DIR)
print("Classifier results:", BASE_DIR / "classifier_comparison.csv")
print("Regression results:", BASE_DIR / "regression_metrics.csv")
print("Missing-value report:", BASE_DIR / "missing_value_report.csv")
print("Best pipeline:", MODEL_PATH)
