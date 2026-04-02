# Data Processing & Model Building Toolkit
This repo contains all the functions needed to perform basic EDA & ETL on datasets, as well as functions to aid in feature engineering, data preprocessing, and model selection.

## Modules & Classes

### 1. Exploratory Data Analysis (`eda_utils.eda.EDA`)
Provides summary statistics and visualization capabilities for pandas dataframes.
- `count_na(col)`: Counts the number of missing values in a pandas column.
- `percentile_25(col)`: Calculates the 25th percentile of a pandas column.
- `percentile_75(col)`: Calculates the 75th percentile of a pandas column.
- `create_data_summary(df)`: Generates a table containing unique counts, missing counts, min, max, mean, median, standard deviation, mode, and percentiles for each feature.
- `dist_plot(df, cols, bin_num)`: Plots histogram distributions for numerical variables in a grid.
- `count_plot(df, cols, split_by, order_by)`: Plots bar counts for categorical variables.
- `box_plots(df, cols)`: Creates box-plots for a list of numeric columns.
- `bivar_box_plt(df, cat_cols, num_cols)`: Generates a grid of box-plots for all combinations of numeric and categorical columns.
- `bivar_line_plot(df, num_cols)`: Generates line plots between pairs of numeric variables to identify relationships.
- `correlation(data, fig_size)`: Plots the correlation heatmap matrix for the given data.
- `xy_scatter_plot(x_data, num_cols, y_data)`: Generates a grid of scatter plots between independent numerical variables and the target variable.

### 2. Feature Engineering (`fe_utils.feature_engineering.FeatureEngineering`)
Utility functions to engineer new features.
- `extract_substr(var_data, start, end, dtype)`: Extracts portions of strings in a pandas Series.
- `create_flags(df, flag_map)`: Converts continuous numerical variables into binary flags based on dictionary thresholds.
- `apply_grouping(data, grp_config, default_group)`: Groups multiple categorical values into newly specified broader categories.

### 3. Data Preprocessing (`fe_utils.pre_processing.DataPreprocessing`)
Fit and transform methods for systematic, reproducible data cleaning and scaling across train/test sets.
- `outlier_fit` / `outlier_transform`: Analyzes and caps outliers using the 1.5 * IQR standard limits.
- `missing_val_fit` / `missing_val_transform`: Derives replacement values (e.g., median/mode) and fills missing data.
- `std_scaler_fit` / `std_scaler_transform`: Scales configurations using sklearn's Standard Scaler.
- `label_enc_fit`: Fits mappings for categorical variables.

### 4. Model Selection (`model_utils.model_selection.ModelSelection`)
Tools to evaluate machine learning models.
- `train_test_compare(...)`: Compares and returns a DataFrame containing Train and Test model performance given standard classification/regression metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC). Supports Binary and Multiclass tasks.
- `choose_best_model(metrics_df, eval_metric)`: Identifies the best performing model from a metrics dataframe based on a specific evaluation metric.
- `model_coeff_linear(model_obj)`: Extracts and returns the feature coefficients and intercept for linear models as a DataFrame.
- `get_feature_importances(model_obj_dict)`: Extracts feature importances or coefficients across multiple different models.
- `plot_actual_v_pred(y_actual, y_pred)`: Creates a line plot comparing the sorted actual target values against the predicted values for visualization.

## Getting Started

Here's a quick example of how you can utilize the package in a notebook or script:

```python
import pandas as pd
from model_building.eda_utils.eda import EDA
from model_building.fe_utils.pre_processing import DataPreprocessing
from model_building.model_utils.model_selection import ModelSelection

df = pd.read_csv('data.csv')

# 1. Exploratory Data Analysis
eda = EDA()
summary_df = eda.create_data_summary(df)
eda.dist_plot(df, cols=['age', 'income'])

# 2. Data Preprocessing
dp = DataPreprocessing()
# Fit on data to find median/mode
missing_config = dp.missing_val_fit(df, num_vars=['age'], cat_vars=['gender'])
# Transform to fill missing values
df_clean = dp.missing_val_transform(df, missing_config)

# 3. Compare Models
# Assumes models are already trained
ms = ModelSelection()
models_dict = {'models': [model1, model2], 'model_name': ['Random Forest', 'XGBoost']}
metrics_comparison = ms.train_test_compare(models_dict, X_train, X_test, y_train, y_test, objective='binary')
```
