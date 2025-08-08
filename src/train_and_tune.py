import os, json, joblib, pandas as pd, numpy as np, yaml, os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PARAMS = yaml.safe_load(open("params.yaml"))
SEED = PARAMS["seed"]
TEST_SIZE = PARAMS["test_size"]
SMALL_ROWS = PARAMS["small_run_rows"]
GRID = PARAMS["model"]["grid"]

DATA = "data/staged/data.csv"

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model.joblib")
COLUMNS_PATH = os.path.join(ART_DIR, "columns.json")
FEATURE_INFO_PATH = os.path.join(ART_DIR, "feature_info.json")
METRICS_PATH = os.path.join(ART_DIR, "metrics.json")

TARGET = "y"

def get_model():
    # default RF; grid will tune n_estimators, max_depth, etc.
    base = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    return base

def main():
    os.makedirs(ART_DIR, exist_ok=True)
    df = pd.read_csv(DATA)
    if os.getenv("SMALL_RUN") == "1" and len(df) > SMALL_ROWS:
        df = df.sample(SMALL_ROWS, random_state=SEED).reset_index(drop=True)

    y = (df[TARGET].astype(str).str.lower().str.strip() == "yes").astype(int)
    X = df.drop(columns=[TARGET])

    # Split deterministic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    pipe = Pipeline([("pre", pre), ("clf", get_model())])

    param_grid = {
        "clf__n_estimators": GRID["n_estimators"],
        "clf__max_depth": GRID["max_depth"],
        "clf__min_samples_split": GRID["min_samples_split"],
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="f1")
    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Persist model & metadata
    joblib.dump(gs.best_estimator_, MODEL_PATH)

    # Capture final transformed feature names
    ohe = gs.best_estimator_.named_steps["pre"].named_transformers_["cat"]
    ohe_features = []
    if hasattr(ohe, "get_feature_names_out"):
        ohe_features = ohe.get_feature_names_out(cat_cols).tolist()

    final_cols = ohe_features + num_cols
    with open(COLUMNS_PATH, "w") as f:
        json.dump(final_cols, f, indent=2)

    feat_info = {"categorical": cat_cols, "numeric": num_cols}
    with open(FEATURE_INFO_PATH, "w") as f:
        json.dump(feat_info, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "best_params": gs.best_params_,
                "accuracy": acc,
                "f1": f1,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            },
            f, indent=2
        )

    print("Saved model and metrics.")
    print({"accuracy": acc, "f1": f1, "best_params": gs.best_params_})

if __name__ == "__main__":
    main()
