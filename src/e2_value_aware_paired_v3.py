from pathlib import Path
import argparse
import pandas as pd
import pydicom
from datetime import datetime

DIRECT_TAGS = [
    "PatientID","PatientName","PatientBirthDate","PatientAddress",
    "PatientTelephoneNumbers","AccessionNumber","StudyID",
    "ReferringPhysicianName","PerformingPhysicianName","OperatorsName",
    "InstitutionName","InstitutionAddress","InstitutionalDepartmentName",
    "StationName","DeviceSerialNumber"
]

DATE_TAGS = [
    "StudyDate","SeriesDate","AcquisitionDate","ContentDate",
    "InstanceCreationDate"
]

UID_TAGS = [
    "StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID",
    "FrameOfReferenceUID"
]

UTILITY_TAGS = [
    "PatientSex","PatientAge","Modality","Manufacturer",
    "ManufacturerModelName","SoftwareVersions","BodyPartExamined",
    "Rows","Columns","BitsAllocated","BitsStored",
    "PhotometricInterpretation","PixelRepresentation"
]

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        if v is None:
            return ""
        return str(v).strip()
    except Exception:
        return ""

def date_diff_days(a, b):
    try:
        da = datetime.strptime(a[:8], "%Y%m%d")
        db = datetime.strptime(b[:8], "%Y%m%d")
        return (db-da).days
    except Exception:
        return None

def read_index(folder):
    rows = []
    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)
            rows.append({
                "file": str(fp),
                "PatientID": sval(ds, "PatientID"),
                "SOPInstanceUID": sval(ds, "SOPInstanceUID"),
                "dataset_obj": ds
            })
        except Exception as e:
            rows.append({
                "file": str(fp),
                "PatientID": "",
                "SOPInstanceUID": "",
                "dataset_obj": None,
                "read_error": str(e)
            })
    return rows

def main():
    ap = argparse.ArgumentParser(description="DICOM-Sentinel E2-v3 paired value-aware analysis")
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--deidentified", required=True)
    ap.add_argument("--patient-crosswalk", required=True)
    ap.add_argument("--uid-crosswalk", required=True)
    ap.add_argument("--output", default="results/e2_v3")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(args.patient_crosswalk, dtype=str).fillna("")
    uid = pd.read_csv(args.uid_crosswalk, dtype=str).fillna("")
    pat_map = dict(zip(pat["evaluation"], pat["de-identified"]))
    uid_map = dict(zip(uid["evaluation"], uid["de-identified"]))

    ev = read_index(args.evaluation)
    de = read_index(args.deidentified)

    ev_good = [r for r in ev if r.get("dataset_obj") is not None and r["SOPInstanceUID"]]
    de_good = [r for r in de if r.get("dataset_obj") is not None and r["SOPInstanceUID"]]

    de_by_sop = {r["SOPInstanceUID"]: r for r in de_good}

    pairs = []
    element_rows = []
    date_rows = []

    for e in ev_good:
        ev_sop = e["SOPInstanceUID"]
        expected_de_sop = uid_map.get(ev_sop, "")
        d = de_by_sop.get(expected_de_sop)

        if d is None:
            pairs.append({
                "evaluation_file": e["file"],
                "evaluation_sop_uid": ev_sop,
                "expected_deidentified_sop_uid": expected_de_sop,
                "matched": False
            })
            continue

        eds = e["dataset_obj"]
        dds = d["dataset_obj"]
        ev_pid = sval(eds, "PatientID")
        de_pid = sval(dds, "PatientID")
        expected_de_pid = pat_map.get(ev_pid, "")

        pairs.append({
            "evaluation_file": e["file"],
            "deidentified_file": d["file"],
            "evaluation_patient_id": ev_pid,
            "deidentified_patient_id": de_pid,
            "expected_deidentified_patient_id": expected_de_pid,
            "patient_mapping_correct": bool(expected_de_pid and de_pid == expected_de_pid),
            "evaluation_sop_uid": ev_sop,
            "deidentified_sop_uid": sval(dds, "SOPInstanceUID"),
            "uid_mapping_correct": sval(dds, "SOPInstanceUID") == expected_de_sop,
            "matched": True
        })

        # Direct identifiers
        for key in DIRECT_TAGS:
            a, b = sval(eds, key), sval(dds, key)

            if not a and not b:
                cls = "absent_both"
            elif a and not b:
                cls = "removed"
            elif not a and b:
                cls = "newly_introduced"
            elif a == b:
                cls = "preserved_same"
            else:
                if key == "PatientID" and expected_de_pid and b == expected_de_pid:
                    cls = "pseudonymized_expected"
                elif key == "PatientName" and expected_de_pid and b.replace("^","") == expected_de_pid.replace("^",""):
                    cls = "pseudonymized_expected"
                else:
                    cls = "transformed"

            element_rows.append({
                "tag_group": "direct_identifier",
                "keyword": key,
                "evaluation_present": bool(a),
                "deidentified_present": bool(b),
                "classification": cls,
                "evaluation_file": e["file"],
                "deidentified_file": d["file"]
            })

        # Dates
        for key in DATE_TAGS:
            a, b = sval(eds, key), sval(dds, key)
            shift = date_diff_days(a,b) if a and b else None

            if not a and not b:
                cls = "absent_both"
            elif a and not b:
                cls = "removed"
            elif not a and b:
                cls = "newly_introduced"
            elif a == b:
                cls = "preserved_same"
            elif shift is not None:
                cls = "date_shifted"
            else:
                cls = "transformed"

            element_rows.append({
                "tag_group": "date",
                "keyword": key,
                "evaluation_present": bool(a),
                "deidentified_present": bool(b),
                "classification": cls,
                "evaluation_file": e["file"],
                "deidentified_file": d["file"]
            })

            if shift is not None:
                date_rows.append({
                    "keyword": key,
                    "shift_days": shift,
                    "evaluation_file": e["file"],
                    "deidentified_file": d["file"]
                })

        # UIDs
        for key in UID_TAGS:
            a, b = sval(eds, key), sval(dds, key)

            if not a and not b:
                cls = "absent_both"
            elif a and not b:
                cls = "removed"
            elif not a and b:
                cls = "newly_introduced"
            elif a == b:
                cls = "preserved_same"
            elif uid_map.get(a, "") == b:
                cls = "uid_remapped_expected"
            else:
                cls = "uid_changed_unverified"

            element_rows.append({
                "tag_group": "uid",
                "keyword": key,
                "evaluation_present": bool(a),
                "deidentified_present": bool(b),
                "classification": cls,
                "evaluation_file": e["file"],
                "deidentified_file": d["file"]
            })

        # Utility / technical tags
        for key in UTILITY_TAGS:
            a, b = sval(eds, key), sval(dds, key)

            if not a and not b:
                cls = "absent_both"
            elif a and not b:
                cls = "removed"
            elif not a and b:
                cls = "newly_introduced"
            elif a == b:
                cls = "preserved_same"
            else:
                cls = "changed"

            element_rows.append({
                "tag_group": "utility",
                "keyword": key,
                "evaluation_present": bool(a),
                "deidentified_present": bool(b),
                "classification": cls,
                "evaluation_file": e["file"],
                "deidentified_file": d["file"]
            })

    pair_df = pd.DataFrame(pairs)
    elem_df = pd.DataFrame(element_rows)
    date_df = pd.DataFrame(date_rows)

    pair_df.to_csv(out/"e2v3_pair_qc.csv", index=False)
    elem_df.to_csv(out/"e2v3_element_classification.csv", index=False)
    date_df.to_csv(out/"e2v3_date_shifts.csv", index=False)

    if not elem_df.empty:
        summary = (
            elem_df.groupby(["tag_group","keyword","classification"])
            .size()
            .reset_index(name="n")
        )
        summary.to_csv(out/"e2v3_classification_summary.csv", index=False)

        unexpected = elem_df[
            (
                (elem_df["tag_group"] == "direct_identifier") &
                (elem_df["classification"] == "preserved_same")
            ) |
            (
                (elem_df["tag_group"] == "uid") &
                (elem_df["classification"].isin(["preserved_same","uid_changed_unverified"]))
            )
        ].copy()
        unexpected.to_csv(out/"e2v3_unexpected_residuals.csv", index=False)

    qc = {
        "evaluation_readable_files": len(ev_good),
        "deidentified_readable_files": len(de_good),
        "paired_files": int(pair_df["matched"].sum()) if not pair_df.empty else 0,
        "unmatched_evaluation_files": int((~pair_df["matched"]).sum()) if not pair_df.empty else 0,
        "patient_mapping_correct_pairs": int(pair_df.get("patient_mapping_correct", pd.Series(dtype=bool)).fillna(False).sum()),
        "uid_mapping_correct_pairs": int(pair_df.get("uid_mapping_correct", pd.Series(dtype=bool)).fillna(False).sum())
    }
    pd.DataFrame([qc]).to_csv(out/"e2v3_qc_summary.csv", index=False)

    print("\nE2-v3 completed")
    print(pd.DataFrame([qc]).to_string(index=False))
    print(f"\nOutputs: {out.resolve()}")

if __name__ == "__main__":
    main()
