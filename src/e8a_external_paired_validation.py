from pathlib import Path
import argparse
import pandas as pd
import pydicom

DIRECT_TAGS = ["PatientID","PatientName","PatientBirthDate","PatientAddress",
"PatientTelephoneNumbers","AccessionNumber","StudyID","ReferringPhysicianName",
"PerformingPhysicianName","OperatorsName","InstitutionName","InstitutionAddress",
"InstitutionalDepartmentName","StationName","DeviceSerialNumber"]

UTILITY_TAGS = ["PatientSex","PatientAge","Modality","Manufacturer","ManufacturerModelName",
"SoftwareVersions","BodyPartExamined","Rows","Columns","SamplesPerPixel",
"PhotometricInterpretation","BitsAllocated","BitsStored","HighBit","PixelRepresentation",
"PixelSpacing","SliceThickness","ImageOrientationPatient","ImagePositionPatient"]

CORE_TAGS = ["SOPClassUID","SOPInstanceUID","StudyInstanceUID","SeriesInstanceUID","Modality"]
UID_TAGS = ["StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID","FrameOfReferenceUID"]

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        return "" if v is None else str(v).strip()
    except Exception:
        return ""

def load_mapping(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    if df.shape[1] < 2:
        raise ValueError(f"Mapping must have >=2 columns: {path}")
    c1, c2 = df.columns[:2]
    return df, dict(zip(df[c1], df[c2])), c1, c2

def read_folder(folder, label):
    rows, failures = [], []
    keys = sorted(set(DIRECT_TAGS + UTILITY_TAGS + CORE_TAGS + UID_TAGS))
    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)
            row = {"dataset": label, "file": str(fp)}
            for k in keys:
                row[k] = sval(ds, k)
            rows.append(row)
        except Exception as e:
            failures.append({"dataset": label, "file": str(fp), "error": str(e)})
    return pd.DataFrame(rows), pd.DataFrame(failures)

def hierarchy(df, label):
    out=[]
    for left,right,name in [
        ("SeriesInstanceUID","StudyInstanceUID","series_to_study"),
        ("SOPInstanceUID","SeriesInstanceUID","sop_to_series"),
        ("SOPInstanceUID","StudyInstanceUID","sop_to_study")]:
        g=df[(df[left]!="")&(df[right]!="")].groupby(left)[right].nunique()
        out.append({"dataset":label,"check":name,"entities_checked":len(g),"violations":int((g>1).sum())})
    return out

def classify(a,b,key,expected_pid=""):
    if not a and not b: return "absent_both"
    if a and not b: return "removed"
    if not a and b: return "newly_introduced"
    if a == b: return "preserved_same"
    if key=="PatientID" and expected_pid and b==expected_pid: return "mapped_expected"
    if key=="PatientName" and expected_pid:
        if b.replace("^","").replace(" ","")==expected_pid.replace("^","").replace(" ",""):
            return "mapped_expected"
    return "transformed"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--synthetic",required=True)
    ap.add_argument("--curated",required=True)
    ap.add_argument("--uid-mapping",required=True)
    ap.add_argument("--patid-mapping",required=True)
    ap.add_argument("--output",default="results/e8a")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    _, uid_map, uid_c1, uid_c2 = load_mapping(args.uid_mapping)
    _, pat_map, pat_c1, pat_c2 = load_mapping(args.patid_mapping)
    syn,sf=read_folder(args.synthetic,"synthetic_validation")
    cur,cf=read_folder(args.curated,"curated_validation")
    if syn.empty or cur.empty: raise SystemExit("No readable DICOM files.")

    cur_by_sop={r["SOPInstanceUID"]:r for _,r in cur.iterrows() if r["SOPInstanceUID"]}
    pairs=[]; direct=[]; utility=[]; uidrows=[]

    for _,sr in syn.iterrows():
        esop=uid_map.get(sr["SOPInstanceUID"],"")
        cr=cur_by_sop.get(esop)
        if cr is None:
            pairs.append({"synthetic_file":sr["file"],"synthetic_sop_uid":sr["SOPInstanceUID"],
                          "expected_curated_sop_uid":esop,"matched":False})
            continue
        epid=pat_map.get(sr["PatientID"],"")
        pairs.append({"synthetic_file":sr["file"],"curated_file":cr["file"],"matched":True,
                      "patient_mapping_correct":bool(epid and cr["PatientID"]==epid),
                      "sop_mapping_correct":cr["SOPInstanceUID"]==esop})
        for k in DIRECT_TAGS:
            direct.append({"keyword":k,"classification":classify(sr.get(k,""),cr.get(k,""),k,epid)})
        for k in UTILITY_TAGS:
            a,b=str(sr.get(k,"") or ""),str(cr.get(k,"") or "")
            cls="absent_both" if not a and not b else "preserved_same" if a==b else "removed" if a and not b else "newly_introduced" if not a and b else "changed"
            utility.append({"keyword":k,"classification":cls})
        for k in UID_TAGS:
            a,b=str(sr.get(k,"") or ""),str(cr.get(k,"") or "")
            cls="absent_both" if not a and not b else "mapped_expected" if a and uid_map.get(a,"")==b else "preserved_same" if a==b else "removed" if a and not b else "newly_introduced" if not a and b else "changed_unverified"
            uidrows.append({"keyword":k,"classification":cls})

    pair_df=pd.DataFrame(pairs)
    pd.DataFrame(direct).groupby(["keyword","classification"]).size().reset_index(name="n").to_csv(out/"e8a_direct_identifier_summary.csv",index=False)
    pd.DataFrame(utility).groupby(["keyword","classification"]).size().reset_index(name="n").to_csv(out/"e8a_utility_summary.csv",index=False)
    pd.DataFrame(uidrows).groupby(["keyword","classification"]).size().reset_index(name="n").to_csv(out/"e8a_uid_summary.csv",index=False)
    pd.DataFrame(hierarchy(syn,"synthetic_validation")+hierarchy(cur,"curated_validation")).to_csv(out/"e8a_hierarchy_audit.csv",index=False)

    presence=[]
    for label,df in [("synthetic_validation",syn),("curated_validation",cur)]:
        for k in CORE_TAGS:
            presence.append({"dataset":label,"tag":k,"present":int((df[k]!="").sum()),"missing":int((df[k]=="").sum()),"missing_rate":float((df[k]=="").mean())})
    pd.DataFrame(presence).to_csv(out/"e8a_core_tag_presence.csv",index=False)

    pd.concat([sf,cf],ignore_index=True).to_csv(out/"e8a_read_failures.csv",index=False)
    pair_df.to_csv(out/"e8a_pair_qc.csv",index=False)
    matched=pair_df[pair_df["matched"]==True]
    qc=pd.DataFrame([{
        "synthetic_readable_files":len(syn),"curated_readable_files":len(cur),
        "synthetic_read_failures":len(sf),"curated_read_failures":len(cf),
        "synthetic_unique_patients":syn["PatientID"].nunique(),"curated_unique_patients":cur["PatientID"].nunique(),
        "synthetic_unique_studies":syn["StudyInstanceUID"].nunique(),"curated_unique_studies":cur["StudyInstanceUID"].nunique(),
        "synthetic_unique_series":syn["SeriesInstanceUID"].nunique(),"curated_unique_series":cur["SeriesInstanceUID"].nunique(),
        "paired_instances":len(matched),
        "unmatched_synthetic_instances":int((pair_df["matched"]==False).sum()),
        "patient_mapping_correct_pairs":int(matched["patient_mapping_correct"].fillna(False).sum()) if len(matched) else 0,
        "sop_mapping_correct_pairs":int(matched["sop_mapping_correct"].fillna(False).sum()) if len(matched) else 0,
        "uid_mapping_source_column":uid_c1,"uid_mapping_target_column":uid_c2,
        "patid_mapping_source_column":pat_c1,"patid_mapping_target_column":pat_c2
    }])
    qc.to_csv(out/"e8a_qc_summary.csv",index=False)
    print(qc.to_string(index=False))

if __name__=="__main__":
    main()
