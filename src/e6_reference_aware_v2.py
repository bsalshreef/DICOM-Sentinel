from pathlib import Path
import argparse
import pandas as pd
import pydicom

FIELDS = [
    "PatientID","StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID",
    "StudyDate","Modality","Rows","Columns","ManufacturerModelName"
]

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        return "" if v is None else str(v).strip()
    except Exception:
        return ""

def read_folder(folder):
    rows = []
    failures = []
    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)
            row = {"basename": fp.name, "file": str(fp)}
            for k in FIELDS:
                row[k] = sval(ds, k)
            rows.append(row)
        except Exception as e:
            failures.append({"file": str(fp), "error": str(e)})
    return pd.DataFrame(rows), pd.DataFrame(failures)

def classify_changes(base_row, tam_row):
    changed = []
    for k in FIELDS:
        if str(base_row.get(k,"")) != str(tam_row.get(k,"")):
            changed.append(k)
    return changed

def expected_field(tamper_type):
    return {
        "T1_patient_id_inconsistency": {"PatientID"},
        "T2_study_uid_inconsistency": {"StudyInstanceUID"},
        "T3_series_uid_inconsistency": {"SeriesInstanceUID"},
        "T4_duplicate_sop_uid": {"SOPInstanceUID"},
        "T5_date_inconsistency": {"StudyDate"},
        "T6_modality_conflict": {"Modality"},
        "T7_geometry_metadata_mismatch": {"Rows","Columns"},
        "T8_device_metadata_inconsistency": {"ManufacturerModelName"},
    }.get(tamper_type, set())

def main():
    ap = argparse.ArgumentParser(description="E6-v2 reference-aware controlled tampering evaluation")
    ap.add_argument("--baseline", required=True, help="Original deidentified DICOM folder")
    ap.add_argument("--tampered", required=True, help="E6 tampered_dataset folder")
    ap.add_argument("--manifest", required=True, help="e6_tamper_manifest.csv")
    ap.add_argument("--output", default="results/e6_v2")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base, base_fail = read_folder(args.baseline)
    tamp, tamp_fail = read_folder(args.tampered)
    manifest = pd.read_csv(args.manifest)

    base_by = {r["basename"]: r for _, r in base.iterrows()}
    tamp_by = {r["basename"]: r for _, r in tamp.iterrows()}

    target_names = {Path(p).name for p in manifest["file"].astype(str)}
    manifest["basename"] = manifest["file"].astype(str).map(lambda x: Path(x).name)

    detail = []
    for name, tr in tamp_by.items():
        br = base_by.get(name)
        if br is None:
            detail.append({
                "basename": name,
                "is_tampered_target": name in target_names,
                "baseline_match_found": False,
                "changed_fields": "",
                "detected_any_change": False
            })
            continue

        changed = classify_changes(br, tr)
        detail.append({
            "basename": name,
            "is_tampered_target": name in target_names,
            "baseline_match_found": True,
            "changed_fields": "|".join(changed),
            "detected_any_change": len(changed) > 0
        })

    detail = pd.DataFrame(detail)

    # Per tamper event/type
    eval_rows = []
    for _, m in manifest.iterrows():
        name = m["basename"]
        d = detail[detail["basename"] == name]
        if d.empty:
            detected = False
            changed = set()
        else:
            changed = set(str(d.iloc[0]["changed_fields"]).split("|")) if d.iloc[0]["changed_fields"] else set()
            expected = expected_field(m["tamper_type"])
            detected = len(changed.intersection(expected)) > 0

        eval_rows.append({
            "basename": name,
            "tamper_type": m["tamper_type"],
            "expected_fields": "|".join(sorted(expected_field(m["tamper_type"]))),
            "observed_changed_fields": "|".join(sorted(changed)),
            "tamper_detected": detected
        })

    event = pd.DataFrame(eval_rows)
    summary = (
        event.groupby("tamper_type")
        .agg(n_tampered=("tamper_detected","size"),
             detected=("tamper_detected","sum"))
        .reset_index()
    )
    summary["sensitivity"] = summary["detected"] / summary["n_tampered"]

    # Controls = untouched files only; any metadata change vs baseline is a true FP.
    controls = detail[~detail["is_tampered_target"]].copy()
    fp = int(controls["detected_any_change"].sum())
    tn = int((~controls["detected_any_change"]).sum())

    # Dataset-level duplicate SOP event detection
    sop_counts = tamp["SOPInstanceUID"].value_counts()
    duplicate_sops = set(sop_counts[sop_counts > 1].index)
    t4 = event[event["tamper_type"] == "T4_duplicate_sop_uid"].copy()
    duplicate_event_detected = 0
    for name in t4["basename"]:
        rr = tamp[tamp["basename"] == name]
        if not rr.empty and rr.iloc[0]["SOPInstanceUID"] in duplicate_sops:
            duplicate_event_detected += 1

    qc = pd.DataFrame([{
        "baseline_files": len(base),
        "tampered_dataset_files": len(tamp),
        "tampered_targets": len(event),
        "untouched_controls": len(controls),
        "control_false_positives": fp,
        "control_true_negatives": tn,
        "control_false_positive_rate": fp/len(controls) if len(controls) else None,
        "baseline_read_failures": len(base_fail),
        "tampered_read_failures": len(tamp_fail),
        "duplicate_sop_tamper_events_detected": duplicate_event_detected,
        "duplicate_sop_tamper_events_total": len(t4)
    }])

    detail.to_csv(out/"e6v2_file_level_changes.csv", index=False)
    event.to_csv(out/"e6v2_event_detail.csv", index=False)
    summary.to_csv(out/"e6v2_detection_summary.csv", index=False)
    qc.to_csv(out/"e6v2_qc_summary.csv", index=False)

    print("\nE6-v2 completed")
    print(qc.to_string(index=False))
    print("\nDetection summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
