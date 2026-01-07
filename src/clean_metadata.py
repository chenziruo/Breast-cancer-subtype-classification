import pandas as pd

m = pd.read_csv("data/metadata.csv", index_col=0)
# normalize labels
m["Subtype"] = m["Subtype"].astype(str).str.strip()
mask_bad = (
    m["Subtype"].isna()
    | (m["Subtype"] == "")
    | (m["Subtype"].str.lower().isin(["nan", "none", "nan."]))
)
removed = m[mask_bad]
kept = m[~mask_bad]
print("Total samples:", len(m))
print("Removed samples with missing label:", len(removed))
kept.to_csv("data/metadata_preprocessed.csv")
# ensure results dir
import os

os.makedirs("results", exist_ok=True)
removed.index.to_series().to_csv(
    "results/removed_samples_nan.csv", index=True, header=["sample"]
)
print("Wrote data/metadata_preprocessed.csv and results/removed_samples_nan.csv")
