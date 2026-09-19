from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
import pydicom

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        return "" if v is None else str(v).strip()
    except Exception:
        return ""

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest() if b is not None else ""

def build_index(folder):
    rows = []
    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, force=False)
            rows.append({
                "file": str(fp),
                "SOPInstanceUID": sval(ds, "SOPInstanceUID"),
                "SeriesInstanceUID": sval(ds, "SeriesInstanceUID"),
                "StudyInstanceUID": sval(ds, "StudyInstanceUID"),
                "Modality": sval(ds, "Modality"),
                "BurnedInAnnotation": sval(ds, "BurnedInAnnotation"),
                "RecognizableVisualFeatures": sval(ds, "RecognizableVisualFeatures"),
                "Rows": sval(ds, "Rows"),
                "Columns": sval(ds, "Columns"),
                "PhotometricInterpretation": sval(ds, "PhotometricInterpretation"),
                "PixelDataSHA256": sha256_bytes(getattr(ds, "PixelData", b"")),
                "ds": ds
            })
        except Exception:
            pass
    return rows

def bbox_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple(mins.tolist() + maxs.tolist())

def analyze_pair(eds, dds):
    a = np.asarray(eds.pixel_array)
    b = np.asarray(dds.pixel_array)

    # Handle multiframe/volumetric by flattening spatially while preserving exact diff statistics
    if a.shape != b.shape:
        return {
            "shape_equal": False,
            "shape_eval": str(a.shape),
            "shape_deid": str(b.shape)
        }

    diff = a.astype(np.float64) - b.astype(np.float64)
    mask = diff != 0
    total = mask.size
    changed = int(mask.sum())

    out = {
        "shape_equal": True,
        "shape_eval": str(a.shape),
        "shape_deid": str(b.shape),
        "changed_pixels": changed,
        "total_pixels": total,
        "changed_fraction": changed / total if total else np.nan,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "eval_min": float(np.min(a)),
        "eval_max": float(np.max(a)),
        "deid_min": float(np.min(b)),
        "deid_max": float(np.max(b)),
    }

    if changed == 0:
        out.update({
            "bbox": "",
            "bbox_fraction": 0.0,
            "border_changed_fraction": 0.0,
            "center_changed_fraction": 0.0,
            "dominant_replacement_value": "",
            "dominant_replacement_fraction": 0.0,
            "localized_change": False,
            "border_concentrated": False,
        })
        return out

    # Reduce to last 2 dims for border/localization assessment.
    m = mask
    if m.ndim > 2:
        m2 = np.any(m, axis=tuple(range(m.ndim - 2)))
    else:
        m2 = m

    h, w = m2.shape[-2], m2.shape[-1]
    border_y = max(1, int(round(h * 0.10)))
    border_x = max(1, int(round(w * 0.10)))

    border = np.zeros((h, w), dtype=bool)
    border[:border_y, :] = True
    border[-border_y:, :] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    center = ~border

    changed2 = int(m2.sum())
    border_changed = int((m2 & border).sum())
    center_changed = int((m2 & center).sum())

    bb = bbox_from_mask(m2)
    if bb is not None:
        y0, x0, y1, x1 = bb
        bbox_area = (y1-y0+1) * (x1-x0+1)
        bbox_fraction = bbox_area / (h*w)
        bbox_str = f"{y0},{x0},{y1},{x1}"
    else:
        bbox_fraction = np.nan
        bbox_str = ""

    # Dominant replacement value in deidentified pixels that changed.
    b_changed = b[mask]
    vals, counts = np.unique(b_changed, return_counts=True)
    j = int(np.argmax(counts))
    dominant_value = vals[j]
    dominant_fraction = float(counts[j] / counts.sum())

    out.update({
        "bbox": bbox_str,
        "bbox_fraction": bbox_fraction,
        "border_changed_fraction": border_changed / changed2 if changed2 else np.nan,
        "center_changed_fraction": center_changed / changed2 if changed2 else np.nan,
        "dominant_replacement_value": str(dominant_value),
        "dominant_replacement_fraction": dominant_fraction,
        # Conservative descriptive flags, not causal labels
        "localized_change": bool(bbox_fraction <= 0.25),
        "border_concentrated": bool(border_changed / changed2 >= 0.80) if changed2 else False,
    })

    return out

def main():
    ap = argparse.ArgumentParser(description="E7b characterization of altered pixel pairs")
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--deidentified", required=True)
    ap.add_argument("--uid-crosswalk", required=True)
    ap.add_argument("--e7-pair-detail", required=True)
    ap.add_argument("--output", default="results/e7b")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(args.e7_pair_detail, dtype=str).fillna("")
    altered = pair[
        pair["decoded_array_exact_equal"].astype(str).str.lower().isin(["false","0"])
    ].copy()

    uid = pd.read_csv(args.uid_crosswalk, dtype=str).fillna("")
    uid_map = dict(zip(uid["evaluation"], uid["de-identified"]))

    ev = build_index(args.evaluation)
    de = build_index(args.deidentified)
    ev_by = {r["SOPInstanceUID"]: r for r in ev if r["SOPInstanceUID"]}
    de_by = {r["SOPInstanceUID"]: r for r in de if r["SOPInstanceUID"]}

    rows = []
    for _, r in altered.iterrows():
        ev_sop = r["evaluation_sop_uid"]
        de_sop = r["deidentified_sop_uid"] or uid_map.get(ev_sop, "")
        er = ev_by.get(ev_sop)
        dr = de_by.get(de_sop)

        if er is None or dr is None:
            rows.append({
                "evaluation_sop_uid": ev_sop,
                "deidentified_sop_uid": de_sop,
                "matched": False
            })
            continue

        eds, dds = er["ds"], dr["ds"]
        stats = analyze_pair(eds, dds)

        row = {
            "evaluation_sop_uid": ev_sop,
            "deidentified_sop_uid": de_sop,
            "matched": True,
            "modality": er["Modality"],
            "evaluation_burned_in_annotation": er["BurnedInAnnotation"],
            "deidentified_burned_in_annotation": dr["BurnedInAnnotation"],
            "evaluation_recognizable_visual_features": er["RecognizableVisualFeatures"],
            "deidentified_recognizable_visual_features": dr["RecognizableVisualFeatures"],
            "photometric_interpretation": er["PhotometricInterpretation"],
        }
        row.update(stats)
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(out/"e7b_anomaly_detail.csv", index=False)

    if not detail.empty:
        summary = pd.DataFrame([{
            "altered_pairs": len(detail),
            "matched_pairs": int(detail["matched"].astype(bool).sum()),
            "localized_changes": int(detail.get("localized_change", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "border_concentrated_changes": int(detail.get("border_concentrated", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "median_changed_fraction": pd.to_numeric(detail.get("changed_fraction", pd.Series(dtype=float)), errors="coerce").median(),
            "max_changed_fraction": pd.to_numeric(detail.get("changed_fraction", pd.Series(dtype=float)), errors="coerce").max(),
        }])
    else:
        summary = pd.DataFrame([{
            "altered_pairs": 0,
            "matched_pairs": 0,
            "localized_changes": 0,
            "border_concentrated_changes": 0,
            "median_changed_fraction": np.nan,
            "max_changed_fraction": np.nan,
        }])

    summary.to_csv(out/"e7b_summary.csv", index=False)

    # A cautious interpretation table; no causal claim.
    interp = []
    for _, r in detail.iterrows():
        if not bool(r.get("matched", False)):
            label = "unmatched"
        else:
            loc = bool(r.get("localized_change", False))
            bord = bool(r.get("border_concentrated", False))
            dom = float(r.get("dominant_replacement_fraction", 0) or 0)
            if loc and bord and dom >= 0.5:
                label = "pattern_consistent_with_localized_masking"
            elif loc:
                label = "localized_pixel_edit"
            else:
                label = "distributed_pixel_change"

        interp.append({
            "evaluation_sop_uid": r.get("evaluation_sop_uid",""),
            "modality": r.get("modality",""),
            "descriptive_pattern": label
        })

    pd.DataFrame(interp).to_csv(out/"e7b_pattern_classification.csv", index=False)

    print("\nE7b completed")
    print(summary.to_string(index=False))
    print("\nDetail:")
    if not detail.empty:
        cols = [c for c in [
            "modality","changed_fraction","bbox_fraction",
            "border_changed_fraction","dominant_replacement_fraction",
            "localized_change","border_concentrated"
        ] if c in detail.columns]
        print(detail[cols].to_string(index=False))

if __name__ == "__main__":
    main()
