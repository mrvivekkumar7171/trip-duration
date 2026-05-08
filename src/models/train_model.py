"""
    Find the best model using lazypredict and then use it for hyperparameter tuning with hyperopt.
    Hyperparameter assume you have already selected the best model for you data.
    We will push this model to GCS and also copy in the root folder for Dockerfile to pick.
    Select train_size for tunning such that there are atleast 1000 rows. 
"""
from lazypredict.Supervised import LazyClassifier, LazyRegressor, CLASSIFIERS, REGRESSORS
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import mean_squared_error, accuracy_score
import pathlib, sys, joblib, mlflow, warnings
import mlflow.sklearn
import pandas as pd

warnings.filterwarnings('ignore')

def find_best_model_with_params(X_train, y_train, X_tune, y_tune, X_test, y_test, is_classification):
    """Find Best Model with LazyPredict and then tune it with Hyperopt"""
    
    if is_classification:
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_tune, X_test, y_tune, y_test)
        model_dict = dict(CLASSIFIERS)
    else:
        clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_tune, X_test, y_tune, y_test)
        model_dict = dict(REGRESSORS)

    # Print the top 3 models, get the name of the best model and create actual model class of it.
    best_model_name = models.index[0]
    BestModelClass = model_dict[best_model_name]
    print(f"\nTop 3 Models:\n{models.head(3)}")

    def get_search_space(model_name):
        """Define search spaces for a few top contenders"""

        if model_name in ['RandomForestClassifier', 'ExtraTreesClassifier']:
            return {
                'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
                'max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
                'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
            }
        elif model_name == 'SVC':
            return {
                'C': hp.loguniform('C', -3, 2),
                'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])
            }
        elif model_name == 'LGBMClassifier':
            return {
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
                'max_depth': hp.choice('max_depth', [3, 5, 10])
            }
        elif model_name == 'RandomForestRegressor':
            return {
                "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
                "max_depth": hp.choice("max_depth", [6, 8, 10]),
                "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
            }
        elif model_name == 'XGBRegressor':
            return {
                "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
                "max_depth": hp.choice("max_depth", [6, 8, 10]),
                "learning_rate": hp.uniform("learning_rate", 0.03, 0.3),
            }
        else:
            return {} # Empty space if model isn't defined above

    space = get_search_space(best_model_name)

    if not space:
        print(f"No search space defined for {best_model_name}. Training with default parameters.")
        
        with mlflow.start_run(run_name=f'{best_model_name} Final Model'):
            model = BestModelClass() # Initialize with defaults
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
            else:
                mlflow.log_metric('RMSE', mean_squared_error(y_test, y_pred) ** 0.5)
                
            mlflow.sklearn.log_model(model, 'model')
            
        return model
        
    else:
        print(f"\nStarting Hyperopt tuning for {best_model_name}...")

        def objective(params_in):
            """Objective function for Hyperopt to minimize the loss metric"""
            if 'max_depth' in params_in and params_in['max_depth'] is not None: params_in['max_depth']=int(params_in['max_depth'])
            if 'min_child_weight' in params_in: params_in['min_child_weight']=int(params_in['min_child_weight']) 
            if 'max_delta_step' in params_in: params_in['max_delta_step']=int(params_in['max_delta_step'])

            model = BestModelClass(**params_in)
            
            # Use cross-validation to evaluate the parameters
            if is_classification:
                score = cross_val_score(model, X_tune, y_tune, cv=5, scoring='accuracy').mean()
                loss = 1 - score # Minimize error (1 - accuracy)
            else:
                score = cross_val_score(model, X_tune, y_tune, cv=5, scoring='neg_mean_squared_error').mean()
                loss = -score 
                
            return {'loss': loss, 'status': STATUS_OK}

        with mlflow.start_run(run_name=f'{best_model_name}_Tuning'):
            argmin = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50, # Number of combinations to try
                trials=Trials(),
                verbose=True
            )

        with mlflow.start_run(run_name=f'{best_model_name} Final Model') as run:
            # configure params
            params = space_eval(space, argmin) 
            if 'max_depth' in params and params['max_depth'] is not None: params['max_depth']=int(params['max_depth'])       
            if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
            if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])  
            mlflow.log_params(params)

            # Train final model on the full training set using best parameters
            model = BestModelClass(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
            else:
                mlflow.log_metric('RMSE', mean_squared_error(y_test, y_pred) ** 0.5)
                
            mlflow.sklearn.log_model(model, 'model')
            
        return model

def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/model.joblib")

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + "/models"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("Experiment_Trip_Duration_Prediction")
    TARGET = "trip_duration" # Change this according to your dataset

    train_features = pd.read_csv(data_path + "/train.csv")
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]
    
    # Auto-detect if task is Classification or Regression based on target variable
    is_classification = True if y.dtype == 'object' or y.nunique() < 20 else False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=222)
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=0.001, random_state=222)
    trained_model = find_best_model_with_params(X_train, y_train, X_tune, y_tune, X_test, y_test, is_classification)
    save_model(trained_model, output_path)
    # print(f"Model successfully saved in '{output_path}'.")


if __name__ == "__main__":
    main()