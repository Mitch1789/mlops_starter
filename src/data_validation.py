import json, pandas as pd

INP = "data/staged/data.csv"
OUT = "data/staged/validation_report.json"

EXPECTED_TARGET = "y"

def main():
    df = pd.read_csv(INP)
    report = {}
    report["shape"] = df.shape
    report["columns"] = df.columns.tolist()
    report["target_present"] = EXPECTED_TARGET in df.columns
    report["null_counts"] = df.isnull().sum().to_dict()
    report["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    # simple distribution snapshots
    sample_stats = {}
    for c in df.select_dtypes(include="number").columns:
        sample_stats[c] = {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": float(df[c].mean())
        }
    report["numeric_stats"] = sample_stats

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved to {OUT}")

if __name__ == "__main__":
    main()
