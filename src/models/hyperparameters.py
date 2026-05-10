from hyperopt import hp

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
def get_search_space(model_name):
    """
    Looks up the search space dictionary. 
    Returns an empty dictionary {} if the model is not found.
    """
    return SEARCH_SPACES.get(model_name, {})

if __name__ == '__main__':
    print(get_search_space('RandomForestRegressor'))
    print(get_search_space('UnknownModel'))