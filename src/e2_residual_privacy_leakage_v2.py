from pathlib import Path
import argparse
import pandas as pd

HIGH_RISK_CLASSES = {"direct_identifier","suspicious_free_text"}
QUASI_IDENTIFIER_CLASSES = {"date_time","device_or_site","uid"}

def normalize_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true":True,"false":False}).fillna(False)

def main():
    ap = argparse.ArgumentParser(description="DICOM-Sentinel E2 v2")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--output", default="results/e2")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inventory, low_memory=False)
    required = {"dataset","file","risk_class","value_present"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["value_present"] = normalize_bool(df["value_present"])
    risk = df[(df["value_present"] == True) & (df["risk_class"].fillna("") != "")].copy()

    summary = (
        risk.groupby(["dataset","risk_class"])
        .agg(elements=("risk_class","size"), affected_files=("file","nunique"))
        .reset_index()
    )
    summary.to_csv(out/"e2_risk_class_leakage.csv", index=False)

    rows = []
    for dataset in ["evaluation","deidentified"]:
        d = risk[risk["dataset"] == dataset]
        rows.append({
            "dataset": dataset,
            "high_risk_elements": int(d["risk_class"].isin(HIGH_RISK_CLASSES).sum()),
            "quasi_identifier_elements": int(d["risk_class"].isin(QUASI_IDENTIFIER_CLASSES).sum()),
            "files_with_high_risk": int(d.loc[d["risk_class"].isin(HIGH_RISK_CLASSES),"file"].nunique()),
        })

    s = pd.DataFrame(rows)
    s.to_csv(out/"e2_dataset_privacy_summary.csv", index=False)

    ev = s[s.dataset == "evaluation"].iloc[0]
    de = s[s.dataset == "deidentified"].iloc[0]

    residual = (
        float(de.high_risk_elements) / float(ev.high_risk_elements)
        if ev.high_risk_elements else None
    )
    metrics = pd.DataFrame([{
        "residual_disclosure_rate": residual,
        "high_risk_removal_rate": None if residual is None else 1-residual,
        "note": "Count-based screening metric; exact paired-tag ground-truth analysis should follow using TCIA mapping/answer-key resources."
    }])
    metrics.to_csv(out/"e2_global_metrics.csv", index=False)

    # Tag-level comparison, without assuming a retained tag is automatically a privacy failure.
    tag = (
        risk.groupby(["dataset","keyword","name","vr","risk_class"], dropna=False)
        .agg(occurrences=("file","size"), affected_files=("file","nunique"))
        .reset_index()
    )
    ev_t = tag[tag.dataset=="evaluation"].drop(columns="dataset")
    de_t = tag[tag.dataset=="deidentified"].drop(columns="dataset")
    keys = ["keyword","name","vr","risk_class"]
    paired = ev_t.merge(de_t, on=keys, how="outer", suffixes=("_evaluation","_deidentified")).fillna(0)
    paired.to_csv(out/"e2_tag_level_screening.csv", index=False)

    print(s.to_string(index=False))
    print("\nE2 v2 completed.")

if __name__ == "__main__":
    main()
