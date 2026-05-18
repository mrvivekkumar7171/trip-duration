import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src.logger import logger

SEARCH_SPACES = {
    'LinearRegression': lambda trial: {},
    'DecisionTree': lambda trial: {
       'max_depth': trial.suggest_categorical('max_depth', [6, 8, 10]),
       'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
       'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
    },
    'RandomForest': lambda trial: {
         'n_estimators': trial.suggest_categorical('n_estimators', [5, 8, 10]), 
         'max_depth': trial.suggest_categorical('max_depth', [6, 8, 10]), 
         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
         'max_samples': trial.suggest_float('max_samples', 0.4, 0.8)
    },
    'GradientBoost': lambda trial: {
        'n_estimators' : trial.suggest_categorical('n_estimators', [15,20,30]), 
        'max_depth': trial.suggest_categorical('max_depth', [4,6,8]), 
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'learning_rate': trial.suggest_float('learning_rate',0.01,0.1),
        'verbose': trial.suggest_categorical('verbose', [1])
    },
    'XtremeGradientBoost': lambda trial: {
        'n_estimators' : trial.suggest_categorical('n_estimators', [15,20,30]), 
        'max_depth': trial.suggest_categorical('max_depth', [4,8,10,12]), 
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3),      
        'gamma': trial.suggest_float('gamma', 0.09, 0.4),
        'verbosity': trial.suggest_categorical('verbosity', [1])
    },
    'RandomForestRegressor': lambda trial: {
        "n_estimators": trial.suggest_categorical("n_estimators", [10, 15, 20]),
        "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    'XGBRegressor': lambda trial: {
        "n_estimators": trial.suggest_categorical("n_estimators", [10, 15, 20]),
        "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3),
    },
    'LogisticRegression': lambda trial: {
        'C': trial.suggest_float('C', 1e-5, 100, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
    },
    'AdaBoostClassifier': lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0)
    },
    'KNeighborsClassifier': lambda trial: {
        'n_neighbors': trial.suggest_categorical('n_neighbors', [3, 5, 7, 11]),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
    },
    'BaggingClassifier': lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [10, 50, 100]),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0)
    },
    'BernoulliNB': lambda trial: {
        'alpha': trial.suggest_float('alpha', 0.1, 2.0)
    },
    'LGBMRegressor': lambda trial: {
        'num_leaves': trial.suggest_categorical('num_leaves', [31, 62, 127]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 500])
    },
    'ElasticNet': lambda trial: {
        'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
    },
    'Lasso': lambda trial: {
        'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True)
    },
    'Ridge': lambda trial: {
        'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True)
    },
    'KNeighborsRegressor': lambda trial: {
        'n_neighbors': trial.suggest_categorical('n_neighbors', [3, 5, 7, 11]),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
    },
    'BayesianRidge': lambda trial: {
        'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1e-5, log=True),
        'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-5, log=True)
    },
    'RandomForestClassifier': lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200]),
        'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10])
    },
    'ExtraTreesClassifier': lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200]),
        'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10])
    },
    'SVC': lambda trial: {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    },
    'LGBMClassifier': lambda trial: {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.0, log=True),
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 5, 10])
    }
}

def get_search_space(trial, model_name : str) -> dict:
    """
    Looks up the search space function from the dictionary and executes it with the trial.
    Returns an empty dictionary {} if the model is not found.
    """
    space_func = SEARCH_SPACES.get(model_name)
    
    if space_func:
        return space_func(trial)
    
    return {}

if __name__ == '__main__':
    pass