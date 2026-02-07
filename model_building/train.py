
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("engine_condition_classification")


api = HfApi(token=os.getenv("HF_TOKEN"))
USERNAME = os.getenv("YOUR_USERNAME")


Xtrain_path = f"hf://datasets/{USERNAME}/car-engine-predictive-maintenence/Xtrain.csv"
Xtest_path = f"hf://datasets/{USERNAME}/car-engine-predictive-maintenence/Xtest.csv"
ytrain_path = f"hf://datasets/{USERNAME}/car-engine-predictive-maintenence/ytrain.csv"
ytest_path = f"hf://datasets/{USERNAME}/car-engine-predictive-maintenencen/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


target = "Engine Condition"
features = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp"
]


# Class imbalance ratio
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (normal:faulty): {class_weight:.2f}")

# Bagging setup
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "baggingclassifier__n_estimators": [50, 100],
    "baggingclassifier__max_samples": [0.8],
    "baggingclassifier__max_features": [0.8],
    "baggingclassifier__estimator__max_depth": [5, 10],
    "baggingclassifier__estimator__min_samples_leaf": [10]
}

model_pipeline = make_pipeline(bagging)

# MLflow experiment
mlflow.set_experiment("engine_condition_classification")

with mlflow.start_run(run_name="bagging_fast_run") as run:
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid,
        n_iter=6,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_
    mlflow.log_params({
        k.replace("baggingclassifier__", "").replace("estimator__", "estimator__"): v
        for k, v in search.best_params_.items()
    })
    mlflow.log_metric("best_cv_roc_auc", search.best_score_)

    # Thresholding
    classification_threshold = 0.40
    mlflow.log_param("classification_threshold", classification_threshold)

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(ytest, y_pred_test_proba)
    })

    log_model_results("bagging_fast_run_production", train_report=train_report, test_report=test_report)

    # Save the model locally
    model_path = "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = f"{USERNAME}/car-engine-predictive-maintenence-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_model.joblib",
        path_in_repo="best_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
