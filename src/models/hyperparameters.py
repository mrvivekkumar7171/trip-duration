from hyperopt import hp
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src.logger import logger

SEARCH_SPACES = {
    'LinearRegression':
    {
    },
    'DecisionTree':
    {
       'max_depth': hp.choice('max_depth', [6,8,10 ]),
       'criterion': hp.choice('criterion', ['ginni', 'entropy']),
       'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None])
    },
    'RandomForest':
    {
         'n_estimators' : hp.choice('n_estimators', [5,8,10]), 
         'max_depth': hp.choice('max_depth', [6,8,10 ]), 
         'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
         'max_samples': hp.uniform('max_sample',0.4,0.8)
    },
    'GradientBoost':
    {
        'n_estimators' : hp.choice('n_estimators', [15,20,30]), 
        'max_depth': hp.choice('max_depth', [4,6,8]), 
        'subsample': hp.uniform('max_sample',0.6,0.8),
        'learning_rate': hp.uniform('learning_rate',0.01,0.1),
        'verbose': hp.choice('verbose', [1])
    },
    'XtremeGradientBoost':
    {
        'n_estimators' : hp.choice('n_estimators', [15,20,30]), 
        'max_depth': hp.choice('max_depth', [4,8,10,12]), 
        'subsample': hp.uniform('max_sample', 0.6, 0.8),
        'learning_rate': hp.uniform('learning_rate', 0.03, 0.3),      
        'gamma':hp.uniform('verbose', 0.09, 0.4),
        'verbosity': hp.choice('verbosity', [1])
    },
    'RandomForestRegressor':
    {
        "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
        "max_depth": hp.choice("max_depth", [6, 8, 10]),
        "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
    },
    'XGBRegressor':
    {
        "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
        "max_depth": hp.choice("max_depth", [6, 8, 10]),
        "learning_rate": hp.uniform("learning_rate", 0.03, 0.3),
    },
    'LogisticRegression': 
    {
        'C': hp.loguniform('C', -5, 2),
        'penalty': hp.choice('penalty', ['l1', 'l2']),
        'solver': hp.choice('solver', ['liblinear', 'saga'])
    },
    'AdaBoostClassifier': 
    {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 1.0)
    },
    'KNeighborsClassifier': 
    {
        'n_neighbors': hp.choice('n_neighbors', [3, 5, 7, 11]),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'metric': hp.choice('metric', ['euclidean', 'manhattan'])
    },
    'BaggingClassifier': 
    {
        'n_estimators': hp.choice('n_estimators', [10, 50, 100]),
        'max_samples': hp.uniform('max_samples', 0.5, 1.0)
    },
    'BernoulliNB': 
    {
        'alpha': hp.uniform('alpha', 0.1, 2.0)
    },
    'LGBMRegressor': 
    {
        'num_leaves': hp.choice('num_leaves', [31, 62, 127]),
        'learning_rate': hp.loguniform('learning_rate', -5, -1),
        'n_estimators': hp.choice('n_estimators', [100, 200, 500])
    },
    'ElasticNet': 
    {
        'alpha': hp.loguniform('alpha', -5, 2),
        'l1_ratio': hp.uniform('l1_ratio', 0, 1)
    },
    'Lasso': 
    {
        'alpha': hp.loguniform('alpha', -5, 2)
    },
    'Ridge': 
    {
        'alpha': hp.loguniform('alpha', -5, 2)
    },
    'KNeighborsRegressor': 
    {
        'n_neighbors': hp.choice('n_neighbors', [3, 5, 7, 11]),
        'weights': hp.choice('weights', ['uniform', 'distance'])
    },
    'BayesianRidge': 
    {
        'alpha_1': hp.loguniform('alpha_1', -7, -5),
        'lambda_1': hp.loguniform('lambda_1', -7, -5)
    },
    'RandomForestClassifier':
    {
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
        'max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
    },
    'ExtraTreesClassifier':
    {
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
        'max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
    },
    'SVC':
    {
        'C': hp.loguniform('C', -3, 2),
        'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])
    },
    'LGBMClassifier':
    {
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [3, 5, 10])
    }
}

def get_search_space(model_name : str) -> dict:
    """
    Looks up the search space dictionary. 
    Returns an empty dictionary {} if the model is not found.
    """
    return SEARCH_SPACES.get(model_name, {})

if __name__ == '__main__':
    logger.info(get_search_space('RandomForestRegressor'))
    logger.info(get_search_space('UnknownModel'))