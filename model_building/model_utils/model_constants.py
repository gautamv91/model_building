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