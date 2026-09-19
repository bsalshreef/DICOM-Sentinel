from pathlib import Path
import argparse
import hashlib
import math
import pandas as pd
import numpy as np
import pydicom

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        return "" if v is None else str(v).strip()
    except Exception:
        return ""

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest() if b is not None else ""

def safe_entropy(arr, bins=256):
    try:
        x = np.asarray(arr, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        hist, _ = np.histogram(x, bins=bins)
        p = hist.astype(np.float64)
        p = p[p > 0]
        p /= p.sum()
        return float(-(p * np.log2(p)).sum())
    except Exception:
        return np.nan

def numeric_features(arr):
    x = np.asarray(arr)
    xf = x.astype(np.float64, copy=False)
    finite = xf[np.isfinite(xf)]
    if finite.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "median": np.nan, "p01": np.nan, "p99": np.nan, "entropy": np.nan
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
        "entropy": safe_entropy(finite),
    }

def read_index(folder):
    rows = []
    failures = []

    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, force=False)
            rows.append({
                "file": str(fp),
                "SOPInstanceUID": sval(ds, "SOPInstanceUID"),
                "SeriesInstanceUID": sval(ds, "SeriesInstanceUID"),
                "StudyInstanceUID": sval(ds, "StudyInstanceUID"),
                "Modality": sval(ds, "Modality"),
                "Rows": sval(ds, "Rows"),
                "Columns": sval(ds, "Columns"),
                "NumberOfFrames": sval(ds, "NumberOfFrames"),
                "TransferSyntaxUID": str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "") or ""),
                "PixelDataSHA256": sha256_bytes(getattr(ds, "PixelData", b"")),
                "PixelDataLength": len(getattr(ds, "PixelData", b"")) if hasattr(ds, "PixelData") else 0,
                "dataset_obj": ds
            })
        except Exception as e:
            failures.append({
                "file": str(fp),
                "error": str(e)
            })

    return rows, failures

def compare_pair(ev_row, de_row):
    eds = ev_row["dataset_obj"]
    dds = de_row["dataset_obj"]

    out = {
        "evaluation_file": ev_row["file"],
        "deidentified_file": de_row["file"],
        "evaluation_sop_uid": ev_row["SOPInstanceUID"],
        "deidentified_sop_uid": de_row["SOPInstanceUID"],
        "modality": ev_row["Modality"],
        "raw_pixeldata_hash_equal": (
            ev_row["PixelDataSHA256"] != "" and
            ev_row["PixelDataSHA256"] == de_row["PixelDataSHA256"]
        ),
        "raw_pixeldata_length_equal": (
            ev_row["PixelDataLength"] == de_row["PixelDataLength"]
        ),
        "rows_equal": ev_row["Rows"] == de_row["Rows"],
        "columns_equal": ev_row["Columns"] == de_row["Columns"],
        "frames_equal": ev_row["NumberOfFrames"] == de_row["NumberOfFrames"],
        "transfer_syntax_equal": ev_row["TransferSyntaxUID"] == de_row["TransferSyntaxUID"],
        "decoded_evaluation_ok": False,
        "decoded_deidentified_ok": False,
        "decoded_shape_equal": False,
        "decoded_dtype_equal": False,
        "decoded_array_exact_equal": False,
        "mae": np.nan,
        "rmse": np.nan,
        "max_abs_diff": np.nan,
        "correlation": np.nan,
    }

    try:
        a = eds.pixel_array
        out["decoded_evaluation_ok"] = True
    except Exception as e:
        out["evaluation_decode_error"] = str(e)
        a = None

    try:
        b = dds.pixel_array
        out["decoded_deidentified_ok"] = True
    except Exception as e:
        out["deidentified_decode_error"] = str(e)
        b = None

    if a is None or b is None:
        return out, None

    out["decoded_shape_equal"] = a.shape == b.shape
    out["decoded_dtype_equal"] = str(a.dtype) == str(b.dtype)

    if a.shape != b.shape:
        return out, None

    out["decoded_array_exact_equal"] = bool(np.array_equal(a, b))

    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    diff = af - bf

    out["mae"] = float(np.mean(np.abs(diff)))
    out["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    out["max_abs_diff"] = float(np.max(np.abs(diff)))

    av = af.ravel()
    bv = bf.ravel()

    if av.size > 1 and np.std(av) > 0 and np.std(bv) > 0:
        out["correlation"] = float(np.corrcoef(av, bv)[0, 1])

    fa = numeric_features(a)
    fb = numeric_features(b)

    feat_rows = []
    for k in fa:
        va = fa[k]
        vb = fb[k]
        equal = False
        absdiff = np.nan
        if pd.notna(va) and pd.notna(vb):
            absdiff = abs(va - vb)
            equal = bool(np.isclose(va, vb, rtol=0, atol=1e-12))

        feat_rows.append({
            "evaluation_sop_uid": ev_row["SOPInstanceUID"],
            "deidentified_sop_uid": de_row["SOPInstanceUID"],
            "modality": ev_row["Modality"],
            "feature": k,
            "evaluation_value": va,
            "deidentified_value": vb,
            "absolute_difference": absdiff,
            "exact_numeric_preservation": equal
        })

    return out, feat_rows

def main():
    ap = argparse.ArgumentParser(description="DICOM-Sentinel E7 pixel and analytical utility preservation")
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--deidentified", required=True)
    ap.add_argument("--uid-crosswalk", required=True)
    ap.add_argument("--output", default="results/e7")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    uid = pd.read_csv(args.uid_crosswalk, dtype=str).fillna("")
    uid_map = dict(zip(uid["evaluation"], uid["de-identified"]))

    ev, ev_fail = read_index(args.evaluation)
    de, de_fail = read_index(args.deidentified)

    de_by_sop = {
        r["SOPInstanceUID"]: r
        for r in de
        if r["SOPInstanceUID"]
    }

    pair_rows = []
    feature_rows = []

    for er in ev:
        ev_sop = er["SOPInstanceUID"]
        de_sop = uid_map.get(ev_sop, "")
        dr = de_by_sop.get(de_sop)

        if dr is None:
            pair_rows.append({
                "evaluation_file": er["file"],
                "evaluation_sop_uid": ev_sop,
                "matched": False
            })
            continue

        pair, feats = compare_pair(er, dr)
        pair["matched"] = True
        pair_rows.append(pair)

        if feats:
            feature_rows.extend(feats)

    pair_df = pd.DataFrame(pair_rows)
    feat_df = pd.DataFrame(feature_rows)

    pair_df.to_csv(out/"e7_pair_detail.csv", index=False)
    feat_df.to_csv(out/"e7_feature_detail.csv", index=False)

    matched = pair_df[pair_df["matched"] == True].copy()

    metrics = []
    for col in [
        "raw_pixeldata_hash_equal",
        "raw_pixeldata_length_equal",
        "rows_equal",
        "columns_equal",
        "frames_equal",
        "transfer_syntax_equal",
        "decoded_evaluation_ok",
        "decoded_deidentified_ok",
        "decoded_shape_equal",
        "decoded_dtype_equal",
        "decoded_array_exact_equal",
    ]:
        if col in matched.columns:
            valid = matched[col].dropna()
            metrics.append({
                "metric": col,
                "n_evaluable": len(valid),
                "n_success": int(valid.astype(bool).sum()),
                "rate": float(valid.astype(bool).mean()) if len(valid) else np.nan
            })

    # Numeric difference metrics among decoded pairs
    decoded = matched[
        (matched["decoded_evaluation_ok"] == True) &
        (matched["decoded_deidentified_ok"] == True) &
        (matched["decoded_shape_equal"] == True)
    ].copy()

    for col in ["mae","rmse","max_abs_diff","correlation"]:
        if col in decoded.columns:
            vals = pd.to_numeric(decoded[col], errors="coerce").dropna()
            metrics.append({
                "metric": col,
                "n_evaluable": len(vals),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan
            })

    pd.DataFrame(metrics).to_csv(out/"e7_summary.csv", index=False)

    if not feat_df.empty:
        feat_summary = (
            feat_df.groupby("feature")
            .agg(
                n_evaluable=("exact_numeric_preservation","size"),
                n_exact=("exact_numeric_preservation","sum"),
                max_absolute_difference=("absolute_difference","max"),
                mean_absolute_difference=("absolute_difference","mean")
            )
            .reset_index()
        )
        feat_summary["exact_preservation_rate"] = (
            feat_summary["n_exact"] / feat_summary["n_evaluable"]
        )
    else:
        feat_summary = pd.DataFrame()

    feat_summary.to_csv(out/"e7_feature_summary.csv", index=False)

    failures = pd.DataFrame(
        [{"dataset":"evaluation", **x} for x in ev_fail] +
        [{"dataset":"deidentified", **x} for x in de_fail]
    )
    failures.to_csv(out/"e7_read_failures.csv", index=False)

    qc = pd.DataFrame([{
        "evaluation_files_read": len(ev),
        "deidentified_files_read": len(de),
        "evaluation_read_failures": len(ev_fail),
        "deidentified_read_failures": len(de_fail),
        "paired_instances": int(matched.shape[0]),
        "unmatched_evaluation_instances": int((pair_df["matched"] == False).sum()),
        "decoded_pairs": int(decoded.shape[0]),
        "raw_pixeldata_hash_equal_pairs": int(matched["raw_pixeldata_hash_equal"].sum()) if len(matched) else 0,
        "decoded_array_exact_equal_pairs": int(decoded["decoded_array_exact_equal"].sum()) if len(decoded) else 0
    }])

    qc.to_csv(out/"e7_qc_summary.csv", index=False)

    # Modality-stratified preservation
    if not matched.empty:
        modality_summary = (
            matched.groupby("modality")
            .agg(
                n_pairs=("matched","size"),
                raw_hash_equal=("raw_pixeldata_hash_equal","sum"),
                decoded_ok=("decoded_deidentified_ok","sum"),
                decoded_exact=("decoded_array_exact_equal","sum")
            )
            .reset_index()
        )
        modality_summary["raw_hash_equal_rate"] = modality_summary["raw_hash_equal"] / modality_summary["n_pairs"]
        modality_summary["decoded_exact_rate"] = modality_summary["decoded_exact"] / modality_summary["n_pairs"]
    else:
        modality_summary = pd.DataFrame()

    modality_summary.to_csv(out/"e7_modality_summary.csv", index=False)

    print("\nE7 completed")
    print(qc.to_string(index=False))
    print("\nSummary:")
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__ == "__main__":
    main()
