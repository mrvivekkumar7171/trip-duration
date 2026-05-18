from sklearn.model_selection import train_test_split
import sys, pathlib, joblib, mlflow, yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, mean_squared_error, r2_score, average_precision_score, ConfusionMatrixDisplay, confusion_matrix
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src.logger import logger

def save_importance_plot(model : object, feature_names : list, output_dir : str) -> None:
    """Save feature importance plot ONLY if the model supports it."""
    if not hasattr(model, 'feature_importances_'):
        return

    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")
    
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(10)
    forest_importances.plot.bar(ax=axes)
    
    plot_path = output_dir / "feature_importance.png"
    fig.savefig(plot_path, bbox_inches='tight')
    mlflow.log_artifact(plot_path.as_posix())
    mlflow.log_artifact(__file__) # To log the source code to the mlflow

def main() -> None:

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["visualize"]
    
    model_file = sys.argv[1]
    data_path = pathlib.Path(sys.argv[2])
    
    # Create a temp folder for MLflow artifacts (images)
    output_dir = pathlib.Path(sys.argv[3])
    output_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri("http://34.131.126.24:5000/")
    mlflow.set_experiment("Trip_Duration_Prediction")

    mlflow.autolog(disable=True)

    TARGET = params['target']

    model = joblib.load(model_file)
    validate_df = pd.read_csv(data_path/'validate_processed.csv')
    X_validate = validate_df.drop(TARGET, axis=1)
    y_validate = validate_df[TARGET]

    validate_ml = X_validate.copy()
    validate_ml[TARGET] = y_validate.copy()
    validate_ml = mlflow.data.from_pandas(validate_ml)

    y_pred = model.predict(X_validate)
    is_classification = params['classification']

    # Log to MLflow
    with mlflow.start_run(run_name="Evaluate_Final_Model"):
        if is_classification:
            mlflow.log_metric("test_accuracy", accuracy_score(y_validate, y_pred))
            
            # Add back ROC AUC and Average Precision (requires probabilities)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_validate)
                # Assuming binary classification based on your original code
                if y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                    mlflow.log_metric("roc_auc_score", roc_auc_score(y_validate, y_prob_pos))
                    mlflow.log_metric("average_precision_score", average_precision_score(y_validate, y_prob_pos))
                    mlflow.log_metric("precision_score", precision_score(y_validate, y_pred))
                    mlflow.log_metric("recall_score", recall_score(y_validate, y_pred))

            # Confusion Matrix Plot
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_validate, y_pred, ax=ax)
            cm_path = output_dir / "confusion_matrix.png"
            fig.savefig(cm_path, bbox_inches='tight')
            mlflow.log_artifact(cm_path.as_posix())

            cm = confusion_matrix(y_validate, y_pred)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix Heatmap')
            cm_path = output_dir / "confusion_matrix_heatmap.png"
            plt.savefig(cm_path, bbox_inches='tight')
            mlflow.log_artifact(cm_path.as_posix())

        else:
            rmse = mean_squared_error(y_validate, y_pred) ** 0.5
            r2 = r2_score(y_validate, y_pred)
            mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2})

        # Feature Importance
        save_importance_plot(model, X_validate.columns.tolist(), output_dir)
        mlflow.log_artifact(__file__)
        mlflow.log_input(validate_ml, "validate")
        mlflow.set_tag("target_type", "classification" if is_classification else "regression")
        mlflow.set_tag("author", "Vivek Kumar")

if __name__ == "__main__":
    main()