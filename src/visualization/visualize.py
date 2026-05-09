from sklearn.model_selection import train_test_split
import sys, pathlib, joblib, mlflow, yaml
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

def save_importance_plot(model, feature_names, output_dir):
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

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["visualize"]
    
    model_file = sys.argv[1]
    data_path = pathlib.Path(sys.argv[2])
    
    # Create a temp folder for MLflow artifacts (images)
    output_dir = pathlib.Path(sys.argv[3])
    output_dir.mkdir(exist_ok=True)

    mlflow.set_experiment("Experiment_Trip_Duration_Prediction")
    TARGET = params['target']

    model = joblib.load(model_file)
    train_df = pd.read_csv(data_path / 'train.csv')
    X = train_df.drop(TARGET, axis=1)
    y = train_df[TARGET]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=222)

    y_pred = model.predict(X_test)
    is_classification = params['classification']

    # Log to MLflow
    with mlflow.start_run(run_name="Evaluate_Final_Model"):
        if is_classification:
            mlflow.log_metric("test_accuracy", metrics.accuracy_score(y_test, y_pred))
            
            # Add back ROC AUC and Average Precision (requires probabilities)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                # Assuming binary classification based on your original code
                if y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                    mlflow.log_metric("roc_auc", metrics.roc_auc_score(y_test, y_prob_pos))
                    mlflow.log_metric("avg_prec", metrics.average_precision_score(y_test, y_prob_pos))

            # Confusion Matrix Plot
            fig, ax = plt.subplots()
            metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            cm_path = output_dir / "confusion_matrix.png"
            fig.savefig(cm_path, bbox_inches='tight')
            mlflow.log_artifact(cm_path.as_posix())
            
        else:
            rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5
            r2 = metrics.r2_score(y_test, y_pred)
            mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2})

        # Feature Importance
        save_importance_plot(model, X_test.columns.tolist(), output_dir)

if __name__ == "__main__":
    main()