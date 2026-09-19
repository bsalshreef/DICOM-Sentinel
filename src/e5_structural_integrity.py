from pathlib import Path
import argparse
import pandas as pd
import pydicom

CORE_TAGS = [
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality"
]

IMAGE_TAGS = [
    "Rows",
    "Columns",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation"
]

GEOMETRY_TAGS = [
    "PixelSpacing",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "FrameOfReferenceUID"
]

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        if v is None:
            return ""
        return str(v).strip()
    except Exception:
        return ""

def read_dataset_folder(folder, label):
    rows = []
    failures = []

    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)

            row = {
                "dataset": label,
                "file": str(fp),
            }

            for key in CORE_TAGS + IMAGE_TAGS + GEOMETRY_TAGS:
                row[key] = sval(ds, key)

            rows.append(row)

        except Exception as e:
            failures.append({
                "dataset": label,
                "file": str(fp),
                "error": str(e)
            })

    return pd.DataFrame(rows), pd.DataFrame(failures)

def duplicate_uid_audit(df, dataset_label):
    rows = []
    for key in ["SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID"]:
        nonempty = df[df[key] != ""]
        counts = nonempty[key].value_counts()
        dup = counts[counts > 1]

        if key == "SOPInstanceUID":
            unexpected = int((dup > 1).sum())
        else:
            # Study/Series UIDs are expected to repeat across instances.
            unexpected = 0

        rows.append({
            "dataset": dataset_label,
            "uid_type": key,
            "unique_values": int(nonempty[key].nunique()),
            "repeated_values": int((counts > 1).sum()),
            "unexpected_duplicate_identifiers": unexpected
        })
    return rows

def hierarchy_audit(df, dataset_label):
    results = []

    # each series should belong to exactly one study
    s = (
        df[df["SeriesInstanceUID"] != ""]
        .groupby("SeriesInstanceUID")["StudyInstanceUID"]
        .nunique()
    )
    results.append({
        "dataset": dataset_label,
        "check": "series_maps_to_single_study",
        "violations": int((s > 1).sum()),
        "entities_checked": int(len(s))
    })

    # each SOP instance should belong to exactly one series
    s = (
        df[df["SOPInstanceUID"] != ""]
        .groupby("SOPInstanceUID")["SeriesInstanceUID"]
        .nunique()
    )
    results.append({
        "dataset": dataset_label,
        "check": "sop_maps_to_single_series",
        "violations": int((s > 1).sum()),
        "entities_checked": int(len(s))
    })

    # each SOP instance should belong to exactly one study
    s = (
        df[df["SOPInstanceUID"] != ""]
        .groupby("SOPInstanceUID")["StudyInstanceUID"]
        .nunique()
    )
    results.append({
        "dataset": dataset_label,
        "check": "sop_maps_to_single_study",
        "violations": int((s > 1).sum()),
        "entities_checked": int(len(s))
    })

    return results

def required_presence_audit(df, dataset_label):
    rows = []
    for key in CORE_TAGS:
        missing = int((df[key] == "").sum())
        rows.append({
            "dataset": dataset_label,
            "tag": key,
            "present": int((df[key] != "").sum()),
            "missing": missing,
            "missing_rate": missing / len(df) if len(df) else None
        })
    return rows

def within_series_consistency(df, dataset_label):
    checks = [
        "Modality",
        "Rows",
        "Columns",
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "PixelRepresentation"
    ]

    rows = []

    for key in checks:
        g = (
            df[df["SeriesInstanceUID"] != ""]
            .groupby("SeriesInstanceUID")[key]
            .nunique(dropna=False)
        )

        rows.append({
            "dataset": dataset_label,
            "attribute": key,
            "series_checked": int(len(g)),
            "series_with_multiple_values": int((g > 1).sum())
        })

    return rows

def paired_compare(ev, de, uid_map):
    de_by_sop = {
        r["SOPInstanceUID"]: r
        for _, r in de.iterrows()
        if r["SOPInstanceUID"]
    }

    rows = []

    preserve_tags = [
        "SOPClassUID",
        "Modality",
        "Rows",
        "Columns",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "PixelSpacing",
        "SliceThickness",
        "ImageOrientationPatient",
        "ImagePositionPatient"
    ]

    for _, er in ev.iterrows():
        ev_sop = er["SOPInstanceUID"]
        expected_de_sop = uid_map.get(ev_sop, "")
        dr = de_by_sop.get(expected_de_sop)

        if dr is None:
            rows.append({
                "evaluation_file": er["file"],
                "matched": False
            })
            continue

        row = {
            "evaluation_file": er["file"],
            "deidentified_file": dr["file"],
            "matched": True
        }

        for key in preserve_tags:
            a = str(er.get(key, "") or "")
            b = str(dr.get(key, "") or "")
            row[f"{key}_preserved"] = (a == b)

        # remapped UIDs should be different but map correctly
        row["SOPInstanceUID_remapped_expected"] = (expected_de_sop == dr["SOPInstanceUID"])

        for uid_key in ["StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"]:
            a = str(er.get(uid_key, "") or "")
            b = str(dr.get(uid_key, "") or "")
            if not a and not b:
                row[f"{uid_key}_status"] = "absent_both"
            elif a and uid_map.get(a, "") == b:
                row[f"{uid_key}_status"] = "remapped_expected"
            elif a == b:
                row[f"{uid_key}_status"] = "preserved_same"
            else:
                row[f"{uid_key}_status"] = "changed_unverified"

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_paired(pair_df):
    rows = []

    if pair_df.empty:
        return pd.DataFrame()

    matched = pair_df[pair_df["matched"] == True]

    for col in matched.columns:
        if col.endswith("_preserved"):
            rows.append({
                "metric": col.replace("_preserved", ""),
                "category": "preservation",
                "n_evaluable": int(len(matched)),
                "n_success": int(matched[col].sum()),
                "rate": float(matched[col].mean())
            })

    for col in matched.columns:
        if col.endswith("_status"):
            counts = matched[col].value_counts(dropna=False)
            for status, n in counts.items():
                rows.append({
                    "metric": col.replace("_status", ""),
                    "category": "uid_status",
                    "status": status,
                    "n_evaluable": int(len(matched)),
                    "n_success": int(n),
                    "rate": float(n / len(matched))
                })

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(
        description="DICOM-Sentinel E5 structural and relational integrity audit"
    )
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--deidentified", required=True)
    ap.add_argument("--uid-crosswalk", required=True)
    ap.add_argument("--output", default="results/e5")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ev, ev_fail = read_dataset_folder(args.evaluation, "evaluation")
    de, de_fail = read_dataset_folder(args.deidentified, "deidentified")

    failures = pd.concat([ev_fail, de_fail], ignore_index=True)
    failures.to_csv(out/"e5_read_failures.csv", index=False)

    uid = pd.read_csv(args.uid_crosswalk, dtype=str).fillna("")
    uid_map = dict(zip(uid["evaluation"], uid["de-identified"]))

    # Basic presence
    presence = pd.DataFrame(
        required_presence_audit(ev, "evaluation") +
        required_presence_audit(de, "deidentified")
    )
    presence.to_csv(out/"e5_core_tag_presence.csv", index=False)

    # UID duplication
    dup = pd.DataFrame(
        duplicate_uid_audit(ev, "evaluation") +
        duplicate_uid_audit(de, "deidentified")
    )
    dup.to_csv(out/"e5_uid_audit.csv", index=False)

    # Hierarchy checks
    hierarchy = pd.DataFrame(
        hierarchy_audit(ev, "evaluation") +
        hierarchy_audit(de, "deidentified")
    )
    hierarchy.to_csv(out/"e5_hierarchy_audit.csv", index=False)

    # Within-series consistency
    consistency = pd.DataFrame(
        within_series_consistency(ev, "evaluation") +
        within_series_consistency(de, "deidentified")
    )
    consistency.to_csv(out/"e5_within_series_consistency.csv", index=False)

    # Paired comparison
    pair_df = paired_compare(ev, de, uid_map)
    pair_df.to_csv(out/"e5_paired_detail.csv", index=False)

    paired_summary = summarize_paired(pair_df)
    paired_summary.to_csv(out/"e5_paired_summary.csv", index=False)

    qc = pd.DataFrame([{
        "evaluation_files": len(ev),
        "deidentified_files": len(de),
        "evaluation_read_failures": len(ev_fail),
        "deidentified_read_failures": len(de_fail),
        "paired_instances": int(pair_df["matched"].sum()) if not pair_df.empty else 0,
        "unmatched_evaluation_instances": int((pair_df["matched"] == False).sum()) if not pair_df.empty else 0,
        "evaluation_unique_studies": int(ev["StudyInstanceUID"].nunique()),
        "evaluation_unique_series": int(ev["SeriesInstanceUID"].nunique()),
        "deidentified_unique_studies": int(de["StudyInstanceUID"].nunique()),
        "deidentified_unique_series": int(de["SeriesInstanceUID"].nunique())
    }])
    qc.to_csv(out/"e5_qc_summary.csv", index=False)

    print("\nE5 completed")
    print(qc.to_string(index=False))

if __name__ == "__main__":
    main()
