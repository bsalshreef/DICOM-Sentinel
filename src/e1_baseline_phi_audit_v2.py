from pathlib import Path
import argparse, hashlib, re
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

# Primary privacy categories are mutually exclusive for counting.
DIRECT_PHI_KEYWORDS = {
    "PatientName","PatientID","PatientBirthDate","PatientBirthTime",
    "PatientAddress","PatientTelephoneNumbers","OtherPatientIDs","OtherPatientNames",
    "MedicalRecordLocator","Occupation","AdditionalPatientHistory",
    "ReferringPhysicianName","PerformingPhysicianName","OperatorsName",
    "PhysiciansOfRecord","RequestingPhysician","StudyID","AccessionNumber"
}
DATE_TIME_KEYWORDS = {
    "StudyDate","SeriesDate","AcquisitionDate","ContentDate","InstanceCreationDate",
    "StudyTime","SeriesTime","AcquisitionTime","ContentTime","InstanceCreationTime",
    "AcquisitionDateTime"
}
UID_KEYWORDS = {"StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID","FrameOfReferenceUID"}
DEVICE_OR_SITE_KEYWORDS = {
    "StationName","DeviceSerialNumber","SoftwareVersions","Manufacturer",
    "ManufacturerModelName","InstitutionName","InstitutionAddress",
    "InstitutionalDepartmentName"
}
SUSPICIOUS_TEXT_PATTERNS = [
    r"\bMRN\b", r"\bDOB\b", r"\bPatient\b", r"\bName\b",
    r"\bAddress\b", r"\bPhone\b", r"\bSSN\b", r"@"
]

def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]

def stringify(value):
    try:
        return "" if value is None else str(value).strip()
    except Exception:
        return ""

def suspicious_free_text(value):
    return bool(value) and any(re.search(p, value, flags=re.I) for p in SUSPICIOUS_TEXT_PATTERNS)

def primary_risk_class(keyword, vr, value):
    # Mutually exclusive hierarchy prevents double counting in E2.
    if keyword in DIRECT_PHI_KEYWORDS or vr == "PN":
        return "direct_identifier"
    if vr in {"LO","LT","SH","ST","UT","UC"} and suspicious_free_text(value):
        return "suspicious_free_text"
    if keyword in DATE_TIME_KEYWORDS:
        return "date_time"
    if keyword in UID_KEYWORDS or vr == "UI":
        return "uid"
    if keyword in DEVICE_OR_SITE_KEYWORDS:
        return "device_or_site"
    return ""

def scan_dataset(ds, file_path, dataset_label, keep_preview=False):
    rows = []
    patient_id = stringify(getattr(ds, "PatientID", ""))
    study_uid = stringify(getattr(ds, "StudyInstanceUID", ""))
    series_uid = stringify(getattr(ds, "SeriesInstanceUID", ""))
    sop_uid = stringify(getattr(ds, "SOPInstanceUID", ""))
    modality = stringify(getattr(ds, "Modality", ""))

    for elem in ds.iterall():
        if elem.VR in {"OB","OW","OF","OD","OL","OV","UN"}:
            continue
        value = stringify(elem.value)
        risk = primary_risk_class(elem.keyword or "", elem.VR, value)
        row = {
            "dataset": dataset_label,
            "file": str(file_path),
            "patient_hash": short_hash(patient_id) if patient_id else "",
            "study_uid_hash": short_hash(study_uid) if study_uid else "",
            "series_uid_hash": short_hash(series_uid) if series_uid else "",
            "sop_uid_hash": short_hash(sop_uid) if sop_uid else "",
            "modality": modality,
            "tag": str(elem.tag),
            "keyword": elem.keyword,
            "name": elem.name,
            "vr": elem.VR,
            "value_present": bool(value),
            "value_length": len(value),
            "risk_class": risk,
            "flagged": bool(risk),
            "is_private": elem.tag.is_private,
        }
        if keep_preview:
            row["value_preview"] = value[:250]
        rows.append(row)
    return rows

def collect_dicom_files(folder):
    return [p for p in Path(folder).rglob("*") if p.is_file()]

def process_folder(folder, label, keep_preview=False):
    records, failed = [], []
    files = collect_dicom_files(folder)
    print(f"[{label}] Candidate files: {len(files)}")
    for i, fp in enumerate(files, 1):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)
            records.extend(scan_dataset(ds, fp, label, keep_preview))
        except (InvalidDicomError, PermissionError, OSError, ValueError) as exc:
            failed.append({"dataset": label, "file": str(fp), "error": str(exc)})
        if i % 250 == 0:
            print(f"[{label}] Processed {i}/{len(files)}")
    return records, failed

def dataset_summary(df, label):
    d = df[df["dataset"] == label].copy()
    flagged = d[(d["flagged"] == True) & (d["value_present"] == True)]
    private_present = d[(d["is_private"] == True) & (d["value_present"] == True)]
    return {
        "dataset": label,
        "dicom_files": d["file"].nunique(),
        "unique_patient_hashes": d.loc[d["patient_hash"] != "", "patient_hash"].nunique(),
        "unique_studies": d.loc[d["study_uid_hash"] != "", "study_uid_hash"].nunique(),
        "unique_series": d.loc[d["series_uid_hash"] != "", "series_uid_hash"].nunique(),
        "unique_sop_instances": d.loc[d["sop_uid_hash"] != "", "sop_uid_hash"].nunique(),
        "flagged_elements_with_values": len(flagged),
        "flagged_files": flagged["file"].nunique(),
        "private_elements_with_values": len(private_present),
        "private_tag_files": private_present["file"].nunique(),
    }

def main():
    ap = argparse.ArgumentParser(description="DICOM-Sentinel E1 v2")
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--deidentified", required=True)
    ap.add_argument("--output", default="results/e1")
    ap.add_argument("--keep-preview", action="store_true",
                    help="Include value_preview column. Use only for synthetic-PHI development data.")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    a, af = process_folder(args.evaluation, "evaluation", args.keep_preview)
    b, bf = process_folder(args.deidentified, "deidentified", args.keep_preview)
    df = pd.DataFrame(a+b)

    if df.empty:
        raise SystemExit("No readable DICOM files found. Check dataset paths.")

    df.to_csv(out/"e1_dicom_metadata_inventory.csv", index=False)
    failed = pd.DataFrame(af+bf)
    if not failed.empty:
        failed.to_csv(out/"e1_failed_files.csv", index=False)

    summary = pd.DataFrame([
        dataset_summary(df, "evaluation"),
        dataset_summary(df, "deidentified")
    ])
    summary.to_csv(out/"e1_dataset_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
