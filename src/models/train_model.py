"""
    Find the best model using lazypredict and then use it for hyperparameter tuning with hyperopt.
    Hyperparameter assume you have already selected the best model for you data.
    We will push this model to GCS and also copy in the root folder for Dockerfile to pick.
    Select train_size for tunning such that there are atleast 1000 rows. 

    To Production, there should be 3 test datasets, in which we test our model on on 2nd and 3rd test dataset
    only once after getting 5 best model on 1st test dataset. Also, use recall and accuracy as metrics for 1st
    test dataset, but for 2nd and 3rd test dataset use only roc as metric.
"""
# Temporarily hide MLflow from Python's import system so that LazyPredict will fail to find MLflow and disable tracking gracefully.
# Restore MLflow so the rest of your script can use it normally
import sys
sys.modules['mlflow'] = None
from lazypredict.Supervised import LazyClassifier, LazyRegressor, CLASSIFIERS, REGRESSORS
del sys.modules['mlflow']

import pathlib, sys, joblib, mlflow, yaml, mlflow.sklearn, mlflow.catboost, mlflow.xgboost, optuna, mlflow.lightgbm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from hyperparameters import get_search_space, SEARCH_SPACES
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src.logger import logger

def find_best_model_with_params(X_train : pd.DataFrame, y_train : pd.Series, X_tune : pd.DataFrame, y_tune : pd.Series, X_test : pd.DataFrame, y_test : pd.Series, is_classification : bool, max_evals : int, TARGET : str) -> object:
    """Find Best Model with LazyPredict and then tune it with Hyperopt"""
    
    if is_classification:
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_tune, X_test, y_tune, y_test)
        model_dict = dict(CLASSIFIERS)
    else:
        clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_tune, X_test, y_tune, y_test)
        model_dict = dict(REGRESSORS)

    best_model_name = models.index[0]
    BestModelClass = model_dict[best_model_name]
    logger.info(f"Top 3 Models:\n{models.head(3)}")

    train_ml = X_train.copy()
    train_ml[TARGET] = y_train.copy()
    train_ml = mlflow.data.from_pandas(train_ml)
    tune_ml = X_tune.copy()
    tune_ml[TARGET] = y_tune.copy()
    tune_ml = mlflow.data.from_pandas(tune_ml)
    test_ml = X_test.copy()
    test_ml[TARGET] = y_test.copy()
    test_ml = mlflow.data.from_pandas(test_ml)

    if best_model_name not in SEARCH_SPACES:
        logger.info(f"No search space defined for {best_model_name}. Training with default parameters.")
        
        with mlflow.start_run(run_name=f'{best_model_name} Final Model'):
            model = BestModelClass() # Initialize with defaults
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
            else:
                mlflow.log_metric('RMSE', mean_squared_error(y_test, y_pred) ** 0.5)

            # To log the model in mlflow
            if 'XGB' in best_model_name:
                mlflow.xgboost.log_model(model, 'model')
            elif 'LGBM' in best_model_name:
                mlflow.lightgbm.log_model(model, 'model')
            elif 'CatBoost' in best_model_name:
                mlflow.catboost.log_model(model, 'model')
            else: # Fallback for RandomForest, Ridge, SVC, LogisticRegression, etc.
                mlflow.sklearn.log_model(model, 'model')
            
            mlflow.log_input(train_ml, "train") # Log the training data as an artifact in mlflow
            mlflow.log_input(test_ml, "test")
            mlflow.set_tag("model_type", best_model_name) # To log the model type as tag in mlflow
            mlflow.set_tag("target_type", "classification" if is_classification else "regression")
            mlflow.set_tag("author", "Vivek Kumar")
            # mlflow.log_artifact(__file__)
            
        return model
        
    else:
        logger.info(f"Starting Optuna tuning for {best_model_name}")

        def objective(trial) -> float:
            """Objective function for Optuna to minimize the loss metric"""
            # 1. Fetch parameters dynamically using the trial object
            params_o = get_search_space(trial, best_model_name)
            
            if 'max_depth' in params_o and params_o['max_depth'] is not None: params_o['max_depth']=int(params_o['max_depth'])
            if 'min_child_weight' in params_o: params_o['min_child_weight']=int(params_o['min_child_weight']) 
            if 'max_delta_step' in params_o: params_o['max_delta_step']=int(params_o['max_delta_step'])

            with mlflow.start_run(run_name=f'{best_model_name}_Tuning', nested=True) as child:
                model = BestModelClass(**params_o, random_state=42)
                
                # Use cross-validation to evaluate the parameters
                if is_classification:
                    score = cross_val_score(model, X_tune, y_tune, cv=5, scoring='accuracy').mean()
                    loss = 1.0 - score # Minimize error (1 - accuracy)
                else:
                    score = cross_val_score(model, X_tune, y_tune, cv=5, scoring='neg_mean_squared_error').mean()
                    loss = -score 
                    
                mlflow.log_params(params_o)
                mlflow.log_metric("loss", loss)
                # 2. Return ONLY the float value (loss)
                return loss

        with mlflow.start_run(run_name=f'{best_model_name}_Tuning', nested=True) as parent:
            # 3. Create the study and run optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=max_evals)
            params_in = study.best_params 

            mlflow.log_metric("best_loss", study.best_value)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_input(tune_ml, "tune")
            mlflow.set_tag("model_type", best_model_name)
            mlflow.set_tag("target_type", "classification" if is_classification else "regression")
            mlflow.set_tag("author", "Vivek Kumar")

        with mlflow.start_run(run_name=f'{best_model_name} Final Model') as run:
            # 4. Retrieve best parameters directly from the study object
            if 'max_depth' in params_in and params_in['max_depth'] is not None: params_in['max_depth']=int(params_in['max_depth'])       
            if 'min_child_weight' in params_in: params_in['min_child_weight']=int(params_in['min_child_weight'])
            if 'max_delta_step' in params_in: params_in['max_delta_step']=int(params_in['max_delta_step'])  

            # Train final model on the full training set using best parameters
            model = BestModelClass(**params_in, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
            else:
                mlflow.log_metric('RMSE', mean_squared_error(y_test, y_pred) ** 0.5)
                
            if 'XGB' in best_model_name:
                mlflow.xgboost.log_model(model, 'model', code_paths=[__file__])
            elif 'LGBM' in best_model_name:
                mlflow.lightgbm.log_model(model, 'model', code_paths=[__file__])
            elif 'CatBoost' in best_model_name:
                mlflow.catboost.log_model(model, 'model', code_paths=[__file__])
            else:
                mlflow.sklearn.log_model(model, 'model', code_paths=[__file__])
            
            mlflow.log_metric("best_loss", study.best_value) # output or result
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()}) # input
            mlflow.log_input(train_ml, "train")
            mlflow.log_input(test_ml, "test")            
            mlflow.set_tag("model_type", best_model_name) # To log the model type as tag in mlflow
            mlflow.set_tag("target_type", "classification" if is_classification else "regression")
            mlflow.set_tag("author", "Vivek Kumar")
            # mlflow.log_artifact(__file__)
            
        return model

def save_model(model : object, output_path : str) -> None:
    """
        Save the trained model to the specified output path
    """
    joblib.dump(model, output_path + "/model.joblib")

def main() -> None:
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    data_path = home_dir / sys.argv[1]
    output_path = home_dir / sys.argv[2]
    data_path_str = data_path.as_posix()
    output_path_str = output_path.as_posix()
    pathlib.Path(output_path_str).mkdir(parents=True, exist_ok=True)

    # mlflow.autolog(disable=True) # Disable autologging to have more control over what gets logged in mlflow and when.
    # mlflow.sklearn.autolog(disable=True) # same can be done for specific library like xgboost, xgboost, statsmodels, lightgbm etc.

    mlflow.set_tracking_uri("http://34.131.126.24:5000/")
    mlflow.set_experiment("Trip_Duration_Prediction")
    TARGET = params['target']

    train_features = pd.read_csv(data_path_str + "/train_processed.csv")
    X_train = train_features.drop(TARGET, axis=1)
    y_train = train_features[TARGET]

    test_features = pd.read_csv(data_path_str + "/test_processed.csv")
    X_test = test_features.drop(TARGET, axis=1)
    y_test = test_features[TARGET]

    is_classification = params['classification']

    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=params['train_size'], random_state=params['train_seed'])
    trained_model = find_best_model_with_params(X_train, y_train, X_tune, y_tune, X_test, y_test, is_classification, params['max_evals'], TARGET)
    save_model(trained_model, output_path_str)

if __name__ == "__main__":
    main()