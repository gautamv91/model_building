##### CONSTANTS FILE

CLASSIFIERS = {'log_c': 'LogisticRegression', 'rf_c': 'RandomForestClassifier', 'xgb_c': 'XGBClassifier',
               'lgbm_c': 'LGBMClassifier'}
REGRESSORS = {'ln_r': 'LinearRegression', 'rf_r': 'RandomForestRegressor', 'xgb_r': 'XGBRegressor',
              'lgbm_r': 'LGBMRegressor'}
ALL_MODELS = {**CLASSIFIERS, **REGRESSORS}  # ** unpacks the dictionary into individual items.


## Hyper-parameter Tuning Methods

GRID_SEARCH = 'grid'
RANDOM_SEARCH = 'random'

## Objectives
BINARY = 'binary'
MULTICLASS = 'multi'
REGRESSION = 'regression'

## Metrics
ALL_METRICS = 'all'

## Classification metrics
ALL_METRIC_NAMES_BIN = ['accuracy', 'precision', 'recall', 'f1 score', 'au-roc']

## Regression metrics
ALL_METRIC_NAMES_REG = ['rmse', 'mse', 'mae', 'mape', 'r2']