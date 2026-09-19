from pathlib import Path
import argparse
import pandas as pd
import pydicom

FEATURE_SETS = {
    "A_modality_only": [
        "Modality"
    ],
    "B_device": [
        "Modality",
        "Manufacturer",
        "ManufacturerModelName",
        "SoftwareVersions"
    ],
    "C_clinical_technical": [
        "Modality",
        "PatientAge",
        "PatientSex",
        "BodyPartExamined",
        "Rows",
        "Columns"
    ],
    "D_combined": [
        "Modality",
        "Manufacturer",
        "ManufacturerModelName",
        "SoftwareVersions",
        "PatientAge",
        "PatientSex",
        "BodyPartExamined",
        "Rows",
        "Columns",
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "PixelRepresentation",
        "SeriesInstanceCount"
    ]
}

def sval(ds, key):
    try:
        v = getattr(ds, key, None)
        if v is None:
            return ""
        return str(v).strip()
    except Exception:
        return ""

def load_series(folder):
    """
    Build one row per SeriesInstanceUID using only non-direct,
    retained metadata suitable for controlled linkability analysis.
    """
    rows = []
    for fp in Path(folder).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)
            series_uid = sval(ds, "SeriesInstanceUID")
            if not series_uid:
                continue

            rows.append({
                "file": str(fp),
                "SeriesInstanceUID": series_uid,
                "StudyInstanceUID": sval(ds, "StudyInstanceUID"),
                "Modality": sval(ds, "Modality"),
                "Manufacturer": sval(ds, "Manufacturer"),
                "ManufacturerModelName": sval(ds, "ManufacturerModelName"),
                "SoftwareVersions": sval(ds, "SoftwareVersions"),
                "PatientAge": sval(ds, "PatientAge"),
                "PatientSex": sval(ds, "PatientSex"),
                "BodyPartExamined": sval(ds, "BodyPartExamined"),
                "Rows": sval(ds, "Rows"),
                "Columns": sval(ds, "Columns"),
                "PhotometricInterpretation": sval(ds, "PhotometricInterpretation"),
                "BitsAllocated": sval(ds, "BitsAllocated"),
                "BitsStored": sval(ds, "BitsStored"),
                "PixelRepresentation": sval(ds, "PixelRepresentation"),
            })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # One row per series. Most selected fields should be stable within a series.
    agg = {}
    for c in df.columns:
        if c in {"file", "SeriesInstanceUID"}:
            continue
        agg[c] = lambda x: x.dropna().astype(str).mode().iloc[0] if len(x.dropna()) else ""

    s = (
        df.groupby("SeriesInstanceUID", as_index=False)
          .agg(agg)
    )

    counts = (
        df.groupby("SeriesInstanceUID")
          .size()
          .rename("SeriesInstanceCount")
          .reset_index()
    )

    s = s.merge(counts, on="SeriesInstanceUID", how="left")
    s["SeriesInstanceCount"] = s["SeriesInstanceCount"].astype(str)
    return s

def exact_signature(row, features):
    vals = []
    for f in features:
        v = str(row.get(f, "") or "").strip()
        vals.append(v)
    return "||".join(vals)

def pair_ground_truth(ev, de, uid_map):
    rows = []
    for _, r in ev.iterrows():
        ev_uid = r["SeriesInstanceUID"]
        de_uid = uid_map.get(ev_uid, "")
        rows.append({
            "evaluation_series_uid": ev_uid,
            "expected_deidentified_series_uid": de_uid,
            "has_ground_truth": bool(de_uid)
        })
    return pd.DataFrame(rows)

def deterministic_analysis(ev, de, gt, features, label):
    ev = ev.copy()
    de = de.copy()

    ev["signature"] = ev.apply(lambda r: exact_signature(r, features), axis=1)
    de["signature"] = de.apply(lambda r: exact_signature(r, features), axis=1)

    de_groups = de.groupby("signature")["SeriesInstanceUID"].apply(list).to_dict()

    results = []
    for _, r in ev.iterrows():
        sig = r["signature"]
        candidates = de_groups.get(sig, [])
        expected = gt.loc[
            gt["evaluation_series_uid"] == r["SeriesInstanceUID"],
            "expected_deidentified_series_uid"
        ]
        expected = expected.iloc[0] if len(expected) else ""

        results.append({
            "feature_set": label,
            "evaluation_series_uid": r["SeriesInstanceUID"],
            "expected_deidentified_series_uid": expected,
            "candidate_count": len(candidates),
            "ground_truth_in_candidates": expected in candidates if expected else False,
            "unique_match": len(candidates) == 1,
            "unique_correct_match": len(candidates) == 1 and candidates[0] == expected,
            "signature": sig
        })

    return pd.DataFrame(results)

def similarity_score(ev_row, de_row, features):
    """
    Simple interpretable matching score:
    proportion of comparable non-empty features that exactly match.
    """
    matches = 0
    comparable = 0

    for f in features:
        a = str(ev_row.get(f, "") or "").strip()
        b = str(de_row.get(f, "") or "").strip()

        if not a or not b:
            continue

        comparable += 1
        if a == b:
            matches += 1

    if comparable == 0:
        return 0.0, 0

    return matches / comparable, comparable

def probabilistic_analysis(ev, de, gt, features, label):
    rows = []

    for _, er in ev.iterrows():
        expected_arr = gt.loc[
            gt["evaluation_series_uid"] == er["SeriesInstanceUID"],
            "expected_deidentified_series_uid"
        ]
        expected = expected_arr.iloc[0] if len(expected_arr) else ""

        candidates = []

        for _, dr in de.iterrows():
            score, ncomp = similarity_score(er, dr, features)
            candidates.append({
                "de_uid": dr["SeriesInstanceUID"],
                "score": score,
                "n_comparable": ncomp
            })

        cand = pd.DataFrame(candidates)
        cand = cand.sort_values(
            ["score", "n_comparable", "de_uid"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        if cand.empty:
            continue

        max_score = cand["score"].max()
        best = cand[cand["score"] == max_score]

        true_row = cand[cand["de_uid"] == expected]
        if true_row.empty:
            true_score = None
            strict_better = None
            equal_to_true = None
            min_rank = None
            max_rank = None
        else:
            true_score = float(true_row.iloc[0]["score"])
            strict_better = int((cand["score"] > true_score).sum())
            equal_to_true = int((cand["score"] == true_score).sum())
            min_rank = strict_better + 1
            max_rank = strict_better + equal_to_true

        rows.append({
            "feature_set": label,
            "evaluation_series_uid": er["SeriesInstanceUID"],
            "expected_deidentified_series_uid": expected,
            "true_score": true_score,
            "best_score": float(max_score),
            "best_tie_size": len(best),
            "strict_top1_correct": (
                len(best) == 1 and best.iloc[0]["de_uid"] == expected
            ),
            "ground_truth_in_best_tie": expected in best["de_uid"].tolist(),
            "true_min_rank": min_rank,
            "true_max_rank": max_rank,
            "top3_possible": (
                min_rank is not None and min_rank <= 3
            ),
            "top5_possible": (
                min_rank is not None and min_rank <= 5
            )
        })

    return pd.DataFrame(rows)

def summarize_det(df):
    n = len(df)
    if n == 0:
        return {}
    return {
        "n_series": n,
        "unique_signature_rate": float(df["unique_match"].mean()),
        "unique_correct_linkage_rate": float(df["unique_correct_match"].mean()),
        "ground_truth_candidate_recall": float(df["ground_truth_in_candidates"].mean()),
        "median_candidate_count": float(df["candidate_count"].median()),
        "max_candidate_count": int(df["candidate_count"].max())
    }

def summarize_prob(df):
    n = len(df)
    if n == 0:
        return {}
    return {
        "n_series": n,
        "strict_top1_accuracy": float(df["strict_top1_correct"].mean()),
        "ground_truth_in_best_tie_rate": float(df["ground_truth_in_best_tie"].mean()),
        "top3_possible_rate": float(df["top3_possible"].mean()),
        "top5_possible_rate": float(df["top5_possible"].mean()),
        "median_best_tie_size": float(df["best_tie_size"].median())
    }

def main():
    ap = argparse.ArgumentParser(
        description="DICOM-Sentinel E4 controlled metadata linkability analysis"
    )
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--deidentified", required=True)
    ap.add_argument("--uid-crosswalk", required=True)
    ap.add_argument("--output", default="results/e4")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ev = load_series(args.evaluation)
    de = load_series(args.deidentified)

    if ev.empty or de.empty:
        raise SystemExit("No readable DICOM series found.")

    uid = pd.read_csv(args.uid_crosswalk, dtype=str).fillna("")
    uid_map = dict(zip(uid["evaluation"], uid["de-identified"]))

    gt = pair_ground_truth(ev, de, uid_map)
    gt.to_csv(out/"e4_ground_truth_series_pairs.csv", index=False)

    ev.to_csv(out/"e4_evaluation_series_features.csv", index=False)
    de.to_csv(out/"e4_deidentified_series_features.csv", index=False)

    det_all = []
    prob_all = []
    det_summary_rows = []
    prob_summary_rows = []

    for label, features in FEATURE_SETS.items():
        det = deterministic_analysis(ev, de, gt, features, label)
        prob = probabilistic_analysis(ev, de, gt, features, label)

        det_all.append(det)
        prob_all.append(prob)

        ds = summarize_det(det)
        ds["feature_set"] = label
        det_summary_rows.append(ds)

        ps = summarize_prob(prob)
        ps["feature_set"] = label
        prob_summary_rows.append(ps)

    det_all = pd.concat(det_all, ignore_index=True)
    prob_all = pd.concat(prob_all, ignore_index=True)

    det_all.to_csv(out/"e4_deterministic_detail.csv", index=False)
    prob_all.to_csv(out/"e4_probabilistic_detail.csv", index=False)

    pd.DataFrame(det_summary_rows).to_csv(
        out/"e4_deterministic_summary.csv", index=False
    )

    pd.DataFrame(prob_summary_rows).to_csv(
        out/"e4_probabilistic_summary.csv", index=False
    )

    feature_table = []
    for label, features in FEATURE_SETS.items():
        for f in features:
            feature_table.append({
                "feature_set": label,
                "feature": f
            })

    pd.DataFrame(feature_table).to_csv(
        out/"e4_feature_sets.csv", index=False
    )

    qc = pd.DataFrame([{
        "evaluation_series": len(ev),
        "deidentified_series": len(de),
        "evaluation_series_with_ground_truth": int(gt["has_ground_truth"].sum()),
        "feature_sets_tested": len(FEATURE_SETS)
    }])

    qc.to_csv(out/"e4_qc_summary.csv", index=False)

    print("\nE4 completed")
    print(qc.to_string(index=False))
    print("\nDeterministic summary:")
    print(pd.DataFrame(det_summary_rows).to_string(index=False))
    print("\nProbabilistic summary:")
    print(pd.DataFrame(prob_summary_rows).to_string(index=False))

if __name__ == "__main__":
    main()
