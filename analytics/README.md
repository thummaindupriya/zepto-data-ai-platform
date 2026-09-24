# Titanic Analytics

## Overview

This module performs exploratory data analysis, statistical analysis, predictive classification, class-imbalance analysis, and fare regression using the classic Titanic dataset.

The dataset is loaded from Seaborn exactly once using:

```python
sns.load_dataset("titanic")
```

Immediately after loading, the dataset is saved as titanic.csv so the modeling stage can work from the saved offline fallback.

## Files

* titanic_analysis.py — complete EDA, modeling, class-imbalance analysis, regression, evaluation, and pipeline-saving script
* titanic.csv — offline Titanic dataset fallback
* missing_value_report.csv — missing-value analysis
* classifier_comparison.csv — classifier performance comparison
* imbalance_comparison.csv — baseline, class-weighted, and SMOTE comparison
* regression_metrics.csv — fare regression metrics
* best_classifier_pipeline.joblib — complete fitted preprocessing + classifier pipeline
* analysis_recommendation.txt — final analytics recommendation
* heteroscedasticity_analysis.csv — residual-spread analysis by prediction quartile
* figures/ — generated charts, confusion matrices, and model visualizations

## Missing-Value Strategy

The original dataset contains:

| Column      | Missing | Percentage | Strategy           |
| ----------- | ------: | ---------: | ------------------ |
| age         |     177 |     19.87% | Median imputation  |
| embarked    |       2 |      0.22% | Drop affected rows |
| deck        |     688 |     77.22% | Drop column        |
| embark_town |       2 |      0.22% | Drop affected rows |

The thresholds follow the assignment requirements:

* Below 5%: drop affected rows
* 5%-30%: impute
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

Boolean masking using `&` was used to calculate combined groups.

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

1. `pclass` and `fare`: **-0.548**
2. `sibsp` and `parch`: **0.415**

The negative relationship between class number and fare indicates that lower numerical passenger-class values are associated with higher fares. The positive relationship between siblings/spouses and parents/children indicates that these family-related variables tend to increase together.

## Multivariate Data Story

### 1. Survival by Sex and Passenger Class

The chart compares survival rates across passenger classes separately for males and females. Survival varies jointly with sex and passenger class, with the largest observed survival rates among female passengers in higher classes.

### 2. Age vs Fare by Survival

The scatter plot shows age and fare while distinguishing survival outcomes. Higher fares are concentrated more heavily among higher-class passengers, while survival outcomes occur across different age and fare combinations.

### 3. Fare Distribution by Class and Survival

The boxplot compares fare distributions across passenger classes and survival outcomes. First-class passengers generally paid substantially higher fares, and the fare distributions also differ between survivors and non-survivors.

### 4. Survival Rate by Family Size

Family size was calculated as `sibsp + parch + 1`. The resulting chart shows that survival rates vary across family-size groups, indicating that family composition provides additional context when interpreting passenger outcomes.

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

The split produced:

* Training rows: **712**
* Testing rows: **179**
* Overall survival rate: **0.3838**
* Training survival rate: **0.3834**
* Testing survival rate: **0.3855**

The preprocessing pipeline:

* Median-imputes numeric missing values
* Most-frequent imputes categorical missing values
* One-hot encodes sex and embarked
* Standardizes numeric features
* Fits preprocessing only on training data

Three classifiers were evaluated.

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |   0.8156 |    0.7903 | 0.7101 | 0.7481 | 0.7904 |
| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8287 |

Confusion matrices are saved in the figures/ directory.

A labeled Decision Tree visualization is also generated and saved in the `figures/` directory.

## Class Imbalance Analysis

The survival target is imbalanced, so three Logistic Regression approaches were compared:

1. Baseline Logistic Regression
2. Logistic Regression with class_weight="balanced"
3. Logistic Regression trained with SMOTE applied only to the training data

| Model                        | Accuracy | Precision | Recall |     F1 |    AUC |
| ---------------------------- | -------: | --------: | -----: | -----: | -----: |
| Baseline Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Balanced Logistic Regression |   0.8045 |    0.7297 | 0.7826 | 0.7552 | 0.8464 |
| SMOTE Logistic Regression    |   0.8101 |    0.7397 | 0.7826 | 0.7606 | 0.8414 |

SMOTE was applied only to the training data after the preprocessing transformation, while the test data remained unchanged. Compared with the baseline, both imbalance-handling approaches increased recall, and the SMOTE model produced the highest F1 among the three approaches.

The results are saved in:

```text
imbalance_comparison.csv
```

## Random Forest Hyperparameter Tuning

Random Forest was tuned using GridSearchCV.

The search covered:

* n_estimators
* max_depth
* max_features

The best parameters from the executed run were:

```text
n_estimators = 100
max_depth = 5
max_features = sqrt
```

Best cross-validation F1:

```text
0.7459
```

The tuned Random Forest produced:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8156 |
| Precision | 0.8750 |
| Recall    | 0.6087 |
| F1        | 0.7179 |
| AUC       | 0.8431 |
| OOB Score | 0.8272 |

The Random Forest was configured with oob_score=True, and the resulting OOB score was **0.8272**.

## Classifier Selection

Using test-set F1 as the selection criterion, the Decision Tree produced the highest F1 among the evaluated final classifiers.

Decision Tree results:

* Accuracy: **0.8156**
* Precision: **0.7903**
* Recall: **0.7101**
* F1: **0.7481**
* AUC: **0.7904**

The selection is based on the measured test-set F1 rather than accuracy alone.

## Fare Regression

A multivariate Linear Regression model was used to predict fare from the other available features.

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

The regression explains a moderate portion of fare variation, while the MAE and RMSE show that substantial prediction error remains.

## Heteroscedasticity Analysis

Residual standard deviation was calculated across prediction quartiles:

| Prediction Quartile | Residual Std |
| ------------------- | -----------: |
| Lowest              |       6.9819 |
| Q2                  |      10.1045 |
| Q3                  |      32.0638 |
| Highest             |      44.6133 |

Residual spread increases toward higher predicted fares. This provides evidence of **heteroscedasticity**, meaning that the variability of regression errors is not constant across the prediction range.

The detailed results are saved in:

```text
heteroscedasticity_analysis.csv
```

## Final Analytics Recommendation

The classifier comparison should be interpreted using F1 and AUC together rather than accuracy alone. The Decision Tree had the highest test-set F1 among the evaluated final classifiers, while the imbalance experiment showed that both class weighting and SMOTE improved positive-class recall compared with the baseline. The fare regression achieved R² = **0.3975** and adjusted R² = **0.3655**, with residual analysis providing evidence of heteroscedasticity. These results should be considered together with the EDA findings and the limitations of the Titanic dataset before applying the workflow to new data.

## Saved Pipeline

The complete selected classifier pipeline is saved as:

```text
best_classifier_pipeline.joblib
```

The saved object contains both preprocessing and the final estimator rather than only the classifier.

The script also reloads the saved pipeline using joblib.load() and demonstrates prediction on a raw passenger record.

## How to Run

From the repository root:

```powershell
python analytics/titanic_analysis.py
```

The script generates:

```text
analytics/
├── titanic_analysis.py
├── titanic.csv
├── missing_value_report.csv
├── classifier_comparison.csv
├── imbalance_comparison.csv
├── regression_metrics.csv
├── heteroscedasticity_analysis.csv
├── analysis_recommendation.txt
├── best_classifier_pipeline.joblib
└── figures/
```

## Reproducibility

The first execution requires internet access because Seaborn downloads the Titanic dataset.

The dataset is immediately saved to:

```text
analytics/titanic.csv
```

Subsequent modeling uses the saved CSV and does not call:

```python
sns.load_dataset("titanic")
```

again.
