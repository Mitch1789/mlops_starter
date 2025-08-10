# import os, time, json, joblib, pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# import boto3

# MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
# COLUMNS_PATH = os.getenv("COLUMNS_PATH", "artifacts/columns.json")
# FEATURE_INFO_PATH = os.getenv("FEATURE_INFO_PATH", "artifacts/feature_info.json")

# PUBLISH_CW = os.getenv("PUBLISH_CW", "0") == "1"
# METRICS_NAMESPACE = os.getenv("METRICS_NAMESPACE", "MLOpsStarter")

# app = FastAPI(title="MLOps Starter Inference")

# # Lazy load
# _model = None
# _columns = None
# _featinfo = None

# class Payload(BaseModel):
#     # Accept arbitrary fields; validated at runtime
#     data: dict

# def load_artifacts():
#     global _model, _columns, _featinfo
#     if _model is None:
#         _model = joblib.load(MODEL_PATH)
#     if _columns is None:
#         _columns = json.load(open(COLUMNS_PATH))
#     if _featinfo is None:
#         _featinfo = json.load(open(FEATURE_INFO_PATH))

# def to_frame(d: dict) -> pd.DataFrame:
#     # Build a single-row DataFrame, then one-hot like training
#     df = pd.DataFrame([d])
#     cat = _featinfo.get("categorical", [])
#     num = _featinfo.get("numeric", [])
#     # one-hot for cat columns; unknowns ignored
#     dummies = pd.get_dummies(df[cat], drop_first=False)
#     full = pd.concat([dummies, df[num]], axis=1)
#     # reindex to training columns
#     full = full.reindex(columns=_columns, fill_value=0)
#     return full

# def publish_metric(name, value):
#     if not PUBLISH_CW:
#         return
#     try:
#         cw = boto3.client("cloudwatch")
#         cw.put_metric_data(
#             Namespace=METRICS_NAMESPACE,
#             MetricData=[{
#                 "MetricName": name,
#                 "Value": float(value),
#                 "Unit": "Milliseconds" if "latency" in name else "Count"
#             }]
#         )
#     except Exception as e:
#         print("CloudWatch publish failed:", e)

# @app.post("/predict")
# def predict(payload: Payload):
#     load_artifacts()
#     start = time.time()
#     X = to_frame(payload.data)
#     proba = _model.predict_proba(X)[:,1][0]
#     pred = int(proba >= 0.5)
#     latency_ms = (time.time() - start) * 1000.0
#     publish_metric("inference_latency_ms", latency_ms)
#     publish_metric("predictions", 1)
#     return {"prediction": pred, "probability": proba, "latency_ms": latency_ms}

import os
import time
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import boto3

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
FEATURE_INFO_PATH = os.getenv("FEATURE_INFO_PATH", "artifacts/feature_info.json")

PUBLISH_CW = os.getenv("PUBLISH_CW", "0") == "1"
METRICS_NAMESPACE = os.getenv("METRICS_NAMESPACE", "MLOpsStarter")

app = FastAPI(title="MLOps Starter Inference")

_model = None
_featinfo = None

class Payload(BaseModel):
    data: dict

def load_artifacts():
    global _model, _featinfo
    if _model is None:
        _model = joblib.load(MODEL_PATH)  # Pipeline(preprocessor, clf)
    if _featinfo is None:
        with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
            _featinfo = json.load(f)

def to_raw_frame(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])
    cats = _featinfo.get("categorical", [])
    nums = _featinfo.get("numeric", [])
    for c in cats:
        if c not in df.columns: df[c] = "missing"
        df[c] = df[c].astype(str)
    for c in nums:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df[cats + nums]

def publish_metric(name: str, value):
    if not PUBLISH_CW:
        return
    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=METRICS_NAMESPACE,
            MetricData=[{
                "MetricName": name,
                "Value": float(value),
                "Unit": "Milliseconds" if "latency" in name else "Count"
            }]
        )
    except Exception as e:
        print("CloudWatch publish failed:", e)

@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict")
def predict(payload: Payload):
    load_artifacts()
    start = time.time()
    X = to_raw_frame(payload.data)
    proba = float(_model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    latency_ms = (time.time() - start) * 1000.0
    publish_metric("inference_latency_ms", latency_ms)
    publish_metric("predictions", 1)
    return {"prediction": pred, "probability": proba, "latency_ms": latency_ms}

