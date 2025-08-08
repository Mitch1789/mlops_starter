import os, zipfile, io, requests, pandas as pd

OUT = "data/staged/data.csv"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
CSV_IN_ZIP = "bank-additional/bank-additional-full.csv"  # ; separator

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"Downloading {URL} ...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(CSV_IN_ZIP) as f:
        df = pd.read_csv(f, sep=';')
    # basic clean: strip col names
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with shape={df.shape}")

if __name__ == "__main__":
    main()
