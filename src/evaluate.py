import json, joblib, pandas as pd, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

PARAMS = yaml.safe_load(open("params.yaml"))
SEED = PARAMS["seed"]
TEST_SIZE = PARAMS["test_size"]

DATA = "data/staged/data.csv"
MODEL_PATH = "artifacts/model.joblib"
OUT = "artifacts/eval_report.json"
TARGET = "y"

def main():
    df = pd.read_csv(DATA)
    y = (df[TARGET].astype(str).str.lower().str.strip() == "yes").astype(int)
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    model = joblib.load(MODEL_PATH)

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    out = {"classification_report": report, "roc_auc": auc}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
