# MLOps Starter: Bank Marketing (Term Deposit) — Git + DVC + SageMaker

This repo implements the required pipeline:
- **Git + DVC** for data and artifact versioning
- **data_ingest → data_validation → train_and_tune → evaluate** (via `dvc.yaml`)
- Containerized **FastAPI** inference service
- **CI/CD** with GitHub Actions
- **SageMaker** deployment (custom container)
- Basic **monitoring hooks** (CloudWatch-ready)

> **Dataset**: UCI Bank Marketing (Portugal) — predicts if a client subscribes a term deposit.
> Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

---

## Quickstart

```bash
# 0) Clone and enter
git clone <your-fork-url>
cd mlops_starter

# 1) Python env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) DVC init + S3 remote
dvc init
dvc remote add -d s3remote s3://<your-bucket>/mlops-starter
# Optional: if using non-default endpoints:
# dvc remote modify s3remote endpointurl https://s3.<region>.amazonaws.com

# 3) Reproduce pipeline
dvc repro

# 4) Push artifacts & data to S3
dvc push
```

## Run the inference API locally

```bash
# Train first to produce artifacts/
dvc repro

# Start API
uvicorn inference.predict:app --host 0.0.0.0 --port 8080

# Sample request
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d @inference/sample.json
```

## Docker (local)

```bash
docker build -t mlops-starter:latest .
docker run -p 8080:8080 mlops-starter:latest
```

## CI/CD (GitHub Actions)

- On PR: lint, tests, small “sanity” train (`SMALL_RUN=1`), `dvc repro`.
- On push to `main`: full `dvc repro`, Docker build & push to ECR, SageMaker deploy/update.

Required repo **secrets**:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- `AWS_ACCOUNT_ID`, `ECR_REPO` (e.g., `mlops-starter`)
- `SAGEMAKER_ROLE_ARN` (exec role), `SAGEMAKER_ENDPOINT_NAME` (e.g., `mlops-starter-endpoint`)

## Monitoring hooks

The API measures latency and can optionally publish to **CloudWatch** if you set:
- `METRICS_NAMESPACE` (e.g., `MLOpsStarter`)
- `PUBLISH_CW=1`

## Architecture (Mermaid)

```mermaid
flowchart LR
  A[GitHub Repo] -->|CI| B[Build & Test]
  B -->|Docker push| C[ECR]
  C -->|CD| D[SageMaker Endpoint]
  subgraph DVC Pipeline
    I[data_ingest] --> V[data_validation] --> T[train_and_tune] --> E[evaluate]
  end
  S3[(S3 DVC Remote)] <-->|dvc push/pull| DVC Pipeline
  User -->|/predict| D
```

## Folder layout
```
project-root/
├─ data/
│  ├─ raw/
│  └─ staged/
├─ src/
│  ├─ data_ingest.py
│  ├─ data_validation.py
│  ├─ train_and_tune.py
│  └─ evaluate.py
├─ inference/
│  ├─ predict.py
│  └─ sample.json
├─ scripts/
│  └─ deploy_sagemaker.py
├─ Dockerfile
├─ dvc.yaml
├─ params.yaml
├─ requirements.txt
├─ .github/workflows/ci-cd.yml
├─ tests/
│  └─ test_ingest.py
└─ docs/
   ├─ final_report_template.md
   └─ architecture.mmd
```

## Final Report
Use `docs/final_report_template.md` as a starting point, export to PDF.
