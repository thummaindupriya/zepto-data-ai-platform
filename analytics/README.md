# Titanic Analytics

## Overview

This module performs exploratory data analysis, statistical analysis, predictive classification, and fare regression using the classic Titanic dataset.

The dataset is loaded from Seaborn exactly once using:

```python
sns.load_dataset("titanic")
```

Immediately after loading, the dataset is saved as titanic.csv so the modeling stage can work from the committed offline fallback.

## Files

* titanic_analysis.py — complete EDA, modeling, regression, evaluation, and pipeline-saving script
* titanic.csv — offline Titanic dataset fallback
* missing_value_report.csv — missing-value analysis
* classifier_comparison.csv — classifier performance comparison
* regression_metrics.csv — fare regression metrics
* best_classifier_pipeline.joblib — complete fitted preprocessing + classifier pipeline
* figures/ — generated charts and confusion matrices

## Missing-Value Strategy

The original dataset contains:

| Column        | Missing | Percentage | Strategy           |
| ------------- | ------: | ---------: | ------------------ |
|  age          |     177 |     19.87% | Median imputation  |
|  embarked     |       2 |      0.22% | Drop affected rows |
|  deck         |     688 |     77.22% | Drop column        |
|  embark_town  |       2 |      0.22% | Drop affected rows |

The thresholds follow the assignment requirements:

* Below 5%: drop affected rows
* 5%–30%: impute
* Above 30%: drop the column when appropriate

For modeling, preprocessing is fitted only on the training data using a scikit-learn Pipeline and ColumnTransformer.

## Univariate Analysis

Age and fare were examined using histograms and boxplots.

IQR outlier analysis produced:

* Age outliers: **65**
* Fare outliers: **114**

Fare statistics:

* Mean: **32.10**
* Median: **14.45**
* Mode: **8.05**

Because mean > median > mode, the fare distribution is interpreted as **right-skewed**.

## Bivariate Analysis

Survival rates:

* Male: **18.9%**
* Female: **74.0%**
* 1st class: **62.6%**
* 2nd class: **47.3%**
* 3rd class: **24.2%**

The sex + class breakdown shows substantial differences across groups. Female first-class passengers had a survival rate of approximately **96.7%**, while male third-class passengers had a survival rate of approximately **13.5%**.

Boolean masking using & was also used to calculate combined groups.

## Correlation Analysis

The correlation matrix contains exactly these six columns:

```text
survived
pclass
age
sibsp
parch
fare
```

The two strongest absolute off-diagonal correlations were:

1. pclass and fare: **-0.548**
2. sibsp and parch: **0.415**

The negative relationship between class number and fare indicates that lower numerical passenger-class values are associated with higher fares. The positive relationship between siblings/spouses and parents/children indicates that these family-related variables tend to increase together.

## Multivariate Data Story

### Survival by Sex and Passenger Class

The chart compares survival rates across passenger classes separately for males and females. It shows that survival varied jointly with sex and class rather than being explained by either variable in isolation.

### Age vs Fare by Survival

The scatter plot shows the relationship between age and fare while distinguishing survival outcomes. Higher fares are concentrated more heavily among first-class passengers, while survival patterns are distributed across age and fare combinations.

### Fare Distribution by Class and Survival

The boxplot compares fare distributions across passenger classes and survival outcomes. First-class passengers generally paid substantially higher fares, while fare distributions also differ between survivors and non-survivors.

### Survival Rate by Family Size

Family size was calculated as sibsp + parch + 1. The resulting chart shows that survival rates vary across family-size groups, suggesting that family composition is relevant when interpreting passenger outcomes.

## Exploratory Standardization

Age and fare were standardized using z-score standardization on the cleaned EDA dataset.

Before standardization:

* Age mean: **29.315**
* Age standard deviation: **12.985**
* Fare mean: **32.097**
* Fare standard deviation: **49.698**

After standardization:

* Age mean: approximately **0**
* Age standard deviation: approximately **1**
* Fare mean: approximately **0**
* Fare standard deviation: approximately **1**

This standardization is exploratory and separate from the modeling preprocessing pipeline.

## Classification Modeling

A stratified 80/20 train-test split was used so that the survival-class proportion remained similar in training and testing data.

The preprocessing pipeline:

* Median-imputes numeric missing values
* Most-frequent imputes categorical missing values
* One-hot encodes `sex` and `embarked`
* Standardizes numeric features
* Fits preprocessing only on training data

Three classifiers were evaluated.

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |   0.8156 |    0.7903 | 0.7101 | 0.7481 | 0.7904 |
| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8287 |

## Hyperparameter Tuning

Random Forest was tuned using GridSearchCV.

Best parameters:

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
```

The tuned Random Forest achieved:

* Accuracy: **0.8156**
* Precision: **0.8333**
* Recall: **0.6522**
* F1: **0.7317**
* AUC: **0.8308**

## Classifier Selection

Using test-set F1 as the selection criterion, the Decision Tree produced the highest F1 among the evaluated final models:

**F1 = 0.7481**

Its accuracy was **0.8156**, recall was **0.7101**, and AUC was **0.7904**.

The selection is based on the measured test metrics rather than a single accuracy value.

## Fare Regression

A multivariate linear regression model was used to predict fare from the other available features.

Results:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 20.8977 |
| RMSE        | 30.5328 |
| R²          |  0.3975 |
| Adjusted R² |  0.3655 |

The residual plot is saved as:

```text
figures/fare_regression_residuals.png
```

The residual plot should be inspected visually for changing spread or systematic patterns. The numerical metrics indicate that the regression explains a moderate portion of fare variation, while substantial prediction error remains.

## Saved Pipeline

The complete best classifier pipeline is saved as:

```text
best_classifier_pipeline.joblib
```

The saved object contains both preprocessing and the final estimator, rather than only the classifier.

The script also reloads the saved pipeline using joblib.load() and demonstrates prediction on a raw passenger record.

## How to Run

From the repository root:

```powershell
python analytics/titanic_analysis.py
```

The script generates:

analytics/
├── titanic_analysis.py
├── titanic.csv
├── missing_value_report.csv
├── classifier_comparison.csv
├── regression_metrics.csv
├── best_classifier_pipeline.joblib
└── figures/

## Reproducibility

The first execution requires internet access because Seaborn downloads the Titanic dataset. The dataset is immediately saved to titanic.csv.

Subsequent modeling uses the saved CSV and does not call sns.load_dataset("titanic") again.
