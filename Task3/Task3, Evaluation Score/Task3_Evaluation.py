import os
import re
import numpy as np
import pandas as pd

DATA_ROOT = "/Users/nonny/Downloads/Senior Project 2025/Task3-Voted Dataset"

PRED_BASE = os.path.join(DATA_ROOT, "Evaluation")
OUT_BASE  = os.path.join(DATA_ROOT, "Evaluation", "_VERIFY_OUTPUT")

GT_MAP = {
    "check-in":  os.path.join(DATA_ROOT, "2. Verify - Task 3 - Check-in (Silver) (1).xlsx"),
    "check-out": os.path.join(DATA_ROOT, "1. Verify - Task 3 - Check-out (Silver).xlsx"),
    "price":     os.path.join(DATA_ROOT, "3. Verify - Task 3 - Price (Silver).xlsx"),
    "staff":     os.path.join(DATA_ROOT, "4. Verify - Task 3 - Staff (Silver) (1).xlsx"),
}

MASTER_SUMMARY_XLSX = os.path.join(OUT_BASE, "TASK3_VERIFY_MASTER_SUMMARY.xlsx")

CATEGORIES = [
    "Contrary(P)Body(N)",
    "Contrary(N)Body(P)",
    "Contrary(P)Body(P)",
    "Contrary(N)Body(N)",
]

YES_POSITIVE_SHEETS = {
    "Contrary(P)Body(N)",
    "Contrary(N)Body(P)",
}

NO_POSITIVE_SHEETS = {
    "Contrary(P)Body(P)",
    "Contrary(N)Body(N)",
}

TEST_COLS = ["Test 1", "Test 2", "Test 3"]
ID_CANDIDATES = ["Column1", "ID", "Id", "id", "No", "no", "Index", "index"]

def norm_yn(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "Yes"
    if s in {"no", "n", "false", "0"}:
        return "No"
    return None

def find_id_col(df):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in ID_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        if "id" in str(c).lower():
            return c
    return df.columns[0]

def find_vote_col(df):
    def norm_col(x):
        x = str(x).strip().lower()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^a-z0-9]+", "", x)
        return x

    norm_map = {norm_col(c): c for c in df.columns}

    keys = [
        "vote",
        "contrary",
        "groundtruth",
        "gt",
        "label",
        "ytrue",
        "truth",
        "contrarygt",
    ]
    for k in keys:
        if k in norm_map:
            return norm_map[k]

    for k, original in norm_map.items():
        if "contrary" in k or "vote" in k:
            return original

    return None

def get_positive_label(category):
    if category in YES_POSITIVE_SHEETS:
        return "Yes"
    if category in NO_POSITIVE_SHEETS:
        return "No"
    raise ValueError(f"Unknown category for positive class mapping: {category}")

def classify_verify(gt, pred, positive_label):
    if gt not in {"Yes", "No"} or pred not in {"Yes", "No"}:
        return None

    if gt == positive_label and pred == positive_label:
        return "TP"
    if gt == positive_label and pred != positive_label:
        return "FN"
    if gt != positive_label and pred == positive_label:
        return "FP"
    if gt != positive_label and pred != positive_label:
        return "TN"
    return None

def safe_div0(a, b):
    return (a / b) if b else 0.0

def safe_div_series(num, den):
    num = pd.to_numeric(num, errors="coerce").fillna(0.0).astype(float)
    den = pd.to_numeric(den, errors="coerce").fillna(0.0).astype(float)
    out = np.where(den != 0, num / den, 0.0)
    return pd.Series(out, index=num.index, dtype=float)

def compute_metrics(vote_series, pred_series, positive_label):
    y_true = vote_series.map(norm_yn)
    y_pred = pred_series.map(norm_yn)

    ok = y_true.notna() & y_pred.notna()
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    n = int(ok.sum())

    if n == 0:
        return {
            "N_valid": 0,
            "TP": 0, "TN": 0, "FP": 0, "FN": 0,
            "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Accuracy": 0.0
        }

    negative_label = "No" if positive_label == "Yes" else "Yes"

    TP = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    TN = int(((y_true == negative_label) & (y_pred == negative_label)).sum())
    FP = int(((y_true == negative_label) & (y_pred == positive_label)).sum())
    FN = int(((y_true == positive_label) & (y_pred == negative_label)).sum())

    precision = safe_div0(TP, TP + FP)
    recall    = safe_div0(TP, TP + FN)
    f1        = safe_div0(2 * precision * recall, precision + recall)
    accuracy  = safe_div0(TP + TN, TP + TN + FP + FN)

    return {
        "N_valid": n,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy)
    }

def path_has_pp_folder(path_str):
    parts = [p.lower() for p in os.path.normpath(path_str).split(os.sep)]
    return "pp" in parts

def shot_sort_key(shot_value):
    m = re.match(r"(\d+)\s*shot", str(shot_value).strip().lower())
    return int(m.group(1)) if m else 999

def normalize_test_label(s):
    return str(s).strip().replace(" ", "")

def run_one(GT_XLSX, PRED_XLSX, OUT_ROOT, model, shot, aspect):
    os.makedirs(OUT_ROOT, exist_ok=True)

    gt_xl   = pd.ExcelFile(GT_XLSX)
    pred_xl = pd.ExcelFile(PRED_XLSX)

    restrict_to_pred_ids = (aspect.lower() == "staff") and path_has_pp_folder(PRED_XLSX)

    if restrict_to_pred_ids:
        print(f"[INFO] staff + pp detected -> restrict GT/eval to IDs found in prediction file: {os.path.basename(PRED_XLSX)}")

    gt_maps = {}
    pred_id_sets = {}

    for cat in CATEGORIES:
        if cat not in gt_xl.sheet_names:
            raise ValueError(f"GT missing sheet '{cat}'. Sheets found: {gt_xl.sheet_names}")

        gtdf = pd.read_excel(GT_XLSX, sheet_name=cat)
        id_col = find_id_col(gtdf)
        vote_col = find_vote_col(gtdf)
        if vote_col is None:
            raise ValueError(f"GT sheet '{cat}' has no GT column. Columns: {list(gtdf.columns)}")

        gt_clean = gtdf[[id_col, vote_col]].copy()
        gt_clean.columns = ["ID", "Vote"]
        gt_clean["Vote"] = gt_clean["Vote"].map(norm_yn)
        gt_clean = gt_clean[gt_clean["ID"].notna()]

        if restrict_to_pred_ids and cat in pred_xl.sheet_names:
            pdf_ids = pd.read_excel(PRED_XLSX, sheet_name=cat)
            pid_col = find_id_col(pdf_ids)
            pred_ids_for_cat = set(pdf_ids[pid_col].dropna().tolist())
            pred_id_sets[cat] = pred_ids_for_cat

            print(f"[INFO] {cat} -> prediction unique IDs = {len(pred_ids_for_cat)}")
            gt_clean = gt_clean[gt_clean["ID"].isin(pred_ids_for_cat)]

        gt_maps[cat] = gt_clean.set_index("ID")["Vote"].to_dict()

    all_metrics_rows = []

    for cat in CATEGORIES:
        if cat not in pred_xl.sheet_names:
            print(f"[SKIP] Pred missing sheet '{cat}' in {os.path.basename(PRED_XLSX)}")
            continue

        positive_label = get_positive_label(cat)

        pdf = pd.read_excel(PRED_XLSX, sheet_name=cat)
        pid_col = find_id_col(pdf)

        missing = [c for c in TEST_COLS if c not in pdf.columns]
        if missing:
            raise ValueError(
                f"Pred sheet '{cat}' missing columns: {missing}\n"
                f"Columns found: {list(pdf.columns)}"
            )

        if restrict_to_pred_ids:
            pred_ids_for_cat = pred_id_sets.get(cat, set())
            pdf = pdf[pdf[pid_col].isin(pred_ids_for_cat)].copy()

        out_xlsx = os.path.join(OUT_ROOT, f"{cat}.xlsx")
        summary_rows = []

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for tcol in TEST_COLS:
                tmp = pdf[[pid_col, tcol]].copy()
                tmp.columns = ["ID", "LLM_Output"]
                tmp["LLM_Output"] = tmp["LLM_Output"].map(norm_yn)

                tmp["Vote"] = tmp["ID"].map(gt_maps[cat])
                tmp = tmp[tmp["Vote"].notna()].copy()

                tmp["Verify"] = [
                    classify_verify(gt, pr, positive_label)
                    for gt, pr in zip(tmp["Vote"], tmp["LLM_Output"])
                ]

                tmp = tmp[["ID", "Vote", "LLM_Output", "Verify"]]
                tmp.to_excel(writer, sheet_name=tcol, index=False)

                met = compute_metrics(tmp["Vote"], tmp["LLM_Output"], positive_label)
                met.update({
                    "Model": model,
                    "Shot": shot,
                    "Topic": aspect,
                    "Category": cat,
                    "Test": tcol,
                    "Positive_Label": positive_label,
                })
                summary_rows.append(met)
                all_metrics_rows.append(met)

            summary_df = pd.DataFrame(summary_rows)[
                ["Model","Shot","Topic","Category","Test","Positive_Label",
                 "N_valid","TP","TN","FP","FN","Precision","Recall","F1","Accuracy"]
            ]
            summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)

    if not all_metrics_rows:
        return pd.DataFrame()

    combined_path = os.path.join(OUT_ROOT, "EVAL_SUMMARY.xlsx")
    combined_df = pd.DataFrame(all_metrics_rows)[
        ["Model","Shot","Topic","Category","Test","Positive_Label",
         "N_valid","TP","TN","FP","FN","Precision","Recall","F1","Accuracy"]
    ]

    with pd.ExcelWriter(combined_path, engine="openpyxl") as w:
        combined_df.to_excel(w, sheet_name="ALL", index=False)

        group_cols = ["Model", "Shot", "Topic", "Category", "Positive_Label"]
        tot = combined_df.groupby(group_cols, as_index=False)[
            ["N_valid", "TP", "TN", "FP", "FN"]
        ].sum()

        tot["Precision"] = safe_div_series(tot["TP"], tot["TP"] + tot["FP"])
        tot["Recall"]    = safe_div_series(tot["TP"], tot["TP"] + tot["FN"])
        tot["F1"]        = safe_div_series(2 * tot["Precision"] * tot["Recall"], tot["Precision"] + tot["Recall"])
        tot["Accuracy"]  = safe_div_series(tot["TP"] + tot["TN"], tot["TP"] + tot["TN"] + tot["FP"] + tot["FN"])

        micro_cat = tot[
            ["Model","Shot","Topic","Category","Positive_Label",
             "Precision","Recall","F1","Accuracy",
             "N_valid","TP","TN","FP","FN"]
        ]
        micro_cat.to_excel(w, sheet_name="MICRO_BY_CATEGORY", index=False)

    return combined_df

FILE_RE = re.compile(
    r"^task3_contrary_(?P<model>.+?)_(?P<shot>\d+shot)_(?P<aspect>check-in|check-out|price|staff)\.xlsx$",
    re.IGNORECASE
)

def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    jobs = []
    for root, _, files in os.walk(PRED_BASE):
        for fn in files:
            if not fn.lower().endswith(".xlsx"):
                continue
            m = FILE_RE.match(fn)
            if not m:
                continue

            model  = m.group("model")
            shot   = m.group("shot")
            aspect = m.group("aspect").lower()

            pred_path = os.path.join(root, fn)
            gt_path   = GT_MAP.get(aspect)
            out_root  = os.path.join(OUT_BASE, aspect, model, shot)

            jobs.append((gt_path, pred_path, out_root, model, shot, aspect))

    if not jobs:
        print(f"[WARN] No prediction files matched pattern under: {PRED_BASE}")
        return

    print(f"[INFO] Found {len(jobs)} files to verify.\n")

    ok = 0
    fail = 0
    master_rows = []

    for gt_path, pred_path, out_root, model, shot, aspect in jobs:
        try:
            if gt_path is None:
                raise ValueError(f"No GT mapping for topic='{aspect}'")

            combined_df = run_one(gt_path, pred_path, out_root, model, shot, aspect)

            if not combined_df.empty:
                master_rows.append(combined_df)

            ok += 1
            print(f"[OK] {aspect} | {model} | {shot} -> {out_root}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {pred_path}\n  Reason: {e}\n")

    if master_rows:
        master_df = pd.concat(master_rows, ignore_index=True)

        all_eval_summary = master_df[
            ["Model","Shot","Topic","Category","Test","Positive_Label",
             "N_valid","TP","TN","FP","FN","Precision","Recall","F1","Accuracy"]
        ].copy()

        avg_metrics = (
            master_df
            .groupby(["Model", "Shot", "Topic", "Category", "Positive_Label"], as_index=False)
            .agg({
                "Precision": "mean",
                "Recall": "mean",
                "F1": "mean",
                "Accuracy": "mean",
            })
        )

        best_base = master_df.copy()
        best_base["_shot_order"] = best_base["Shot"].map(shot_sort_key)
        best_base["_test_norm"] = best_base["Test"].map(normalize_test_label)

        best_pick = (
            best_base
            .sort_values(
                by=[
                    "Model", "Shot", "Topic", "Category", "Positive_Label",
                    "F1", "Accuracy", "Recall", "Precision", "_test_norm"
                ],
                ascending=[True, True, True, True, True, False, False, False, False, True]
            )
            .groupby(["Model", "Shot", "Topic", "Category", "Positive_Label"], as_index=False)
            .first()
        )

        best_test_wide = (
            best_pick[["Model", "Shot", "Topic", "Category", "Positive_Label", "Test"]]
            .pivot_table(
                index=["Model", "Topic", "Category", "Positive_Label"],
                columns="Shot",
                values="Test",
                aggfunc="first"
            )
            .reset_index()
        )

        fixed_cols = ["Model", "Topic", "Category", "Positive_Label"]
        shot_cols = [c for c in best_test_wide.columns if c not in fixed_cols]
        shot_cols = sorted(shot_cols, key=shot_sort_key)
        best_test_wide = best_test_wide[fixed_cols + shot_cols].rename(
            columns={c: f"Best Test {c}" for c in shot_cols}
        )

        avg_by_model_shot_topic_cat = avg_metrics.merge(
            best_test_wide,
            on=["Model", "Topic", "Category", "Positive_Label"],
            how="left"
        )

        avg_by_model_shot_topic_cat["_shot_order"] = avg_by_model_shot_topic_cat["Shot"].map(shot_sort_key)
        avg_by_model_shot_topic_cat = (
            avg_by_model_shot_topic_cat
            .sort_values(by=["Model", "_shot_order", "Topic", "Category"])
            .drop(columns="_shot_order")
            .reset_index(drop=True)
        )

        with pd.ExcelWriter(MASTER_SUMMARY_XLSX, engine="openpyxl") as writer:
            all_eval_summary.to_excel(writer, sheet_name="ALL_EVAL_SUMMARY", index=False)
            avg_by_model_shot_topic_cat.to_excel(
                writer,
                sheet_name="AVG_BY_MODEL_SHOT_TOPIC_CAT",
                index=False
            )

        print("\n[MASTER SUMMARY SAVED]")
        print(MASTER_SUMMARY_XLSX)

    print("\n====================")
    print(f"DONE OK={ok}  FAIL={fail}")
    print("Output base:")
    print(OUT_BASE)

if __name__ == "__main__":
    main()