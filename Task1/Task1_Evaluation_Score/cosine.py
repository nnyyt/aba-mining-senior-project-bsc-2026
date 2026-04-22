import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============== Config =================

SRC_XLSX = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
SHEET    = "Sheet2"

TASK1_ROOT = "/Users/nonny/Downloads/Senior Project 2025/Task1"

SHOT_FOLDERS = [f"{i}-shot" for i in range(6)]

OUTPUT_ROOT = os.path.join(TASK1_ROOT, "Task1_cosine_evaluation")

FINAL_EXCEL_PATH = os.path.join(OUTPUT_ROOT, "Task1_cosine_evaluation.xlsx")

SKIP_IDS = {
    2, 18, 21, 29, 33, 37, 44, 51, 65, 66, 69, 84, 85, 93, 104, 111, 116, 119,
    122, 135, 141, 142, 143, 145, 146, 149, 153, 154, 156, 158, 161, 167, 168,
    169, 171, 172, 184, 185, 189, 194, 195, 210, 213, 218, 221, 224, 226, 227,
    228, 245, 250, 253, 255, 258, 261, 265, 269, 276, 280, 283, 289, 290, 292,
    293, 295, 310, 325, 329, 331, 337, 340, 341, 359, 372, 382, 386, 395, 400,
    403, 405, 406, 408, 415, 418, 446, 452, 476, 479, 485, 495, 511, 514, 524,
    525, 539, 542, 556, 572, 574, 578, 580, 582, 589, 590, 594, 595, 596, 599,
    600, 606, 607, 610, 614, 615, 618, 620, 633, 639, 656, 664, 665, 677, 681,
    682, 689, 693, 694, 705, 706, 707, 715, 717, 719, 721, 733, 737, 738, 740,
    741, 749, 751, 760, 761, 762, 771, 772, 773, 774, 775, 778, 779
}


# ============== Helper functions =================

def load_answer_key(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    df_answer_key = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    df_answer_key = df_answer_key[~df_answer_key['Column1'].isin(SKIP_IDS)]

    df_answer_cleaned = df_answer_key[
        ['Column1', 'Topic', 'Selected Content', 'Pos/Neg']
    ].rename(columns={
        'Column1': 'ID',
        'Topic': 'Topics',
        'Selected Content': 'Text',
        'Pos/Neg': 'NegPos'
    })

    return df_answer_cleaned


def clean_and_merge(output_csv: str, df_answer_cleaned: pd.DataFrame) -> pd.DataFrame:
    df_output = pd.read_csv(output_csv)
    df_output = df_output[~df_output['ID'].isin(SKIP_IDS)]

    def clean_topic(topic):
        return re.sub(r'\d+$', '', str(topic)).strip()

    df_output = df_output[['ID', 'Topics', 'Text', 'NegPos']].copy()
    df_output['Topics'] = df_output['Topics'].apply(clean_topic)
    df_output['Text'] = df_output['Text'].astype(str)

    df_output = df_output[df_output['Topics'] != 'Off']

    def join_text(x):
        vals = []
        for v in x:
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            vals.append(s)
        return ", ".join(vals) if vals else ""

    def pick_label(x):
        vals = []
        for v in x:
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            vals.append(s)
        return vals[0] if vals else ""

    df_output_agg = (
        df_output
        .groupby(['ID', 'Topics'])
        .agg(Text=('Text', join_text),
             NegPos=('NegPos', pick_label))
        .reset_index()
    )

    df_ans = df_answer_cleaned.copy()
    df_ans['Topics'] = df_ans['Topics'].astype(str).str.strip()
    df_ans['Text']   = df_ans['Text'].astype(str)
    df_ans['NegPos'] = df_ans['NegPos'].astype(str)

    df_ans = df_ans[df_ans['Topics'] != 'Off']

    df_ans_agg = (
        df_ans
        .groupby(['ID', 'Topics'])
        .agg(Text=('Text', join_text),
             NegPos=('NegPos', pick_label))
        .reset_index()
    )

    ALL_TOPICS = [
        'Room', 'Staff', 'Location', 'Food', 'Price',
        'Facility', 'Check-in', 'Check-out',
        'Taxi-issue', 'Booking-issue'
    ]

    all_ids = sorted(set(df_ans_agg['ID']).union(df_output_agg['ID']))

    base = (
        pd.MultiIndex
        .from_product([all_ids, ALL_TOPICS], names=['ID', 'Topics'])
        .to_frame(index=False)
    )

    df_compare = (
        base
        .merge(
            df_ans_agg.rename(columns={'Text': 'Text_answer', 'NegPos': 'NegPos_answer'}),
            on=['ID', 'Topics'],
            how='left'
        )
        .merge(
            df_output_agg.rename(columns={'Text': 'Text_output', 'NegPos': 'NegPos_output'}),
            on=['ID', 'Topics'],
            how='left'
        )
    )

    for col in ['Text_answer', 'Text_output', 'NegPos_answer', 'NegPos_output']:
        df_compare[col] = df_compare[col].fillna("")

    def compute_cosine(a: str, b: str) -> float:
        a = str(a).strip()
        b = str(b).strip()
        if a == "" or b == "":
            return 0.0
        vect = TfidfVectorizer().fit([a, b])
        tfidf = vect.transform([a, b])
        score = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
        return float(score)

    df_compare['Cosine_Score'] = df_compare.apply(
        lambda row: compute_cosine(row['Text_answer'], row['Text_output']),
        axis=1
    )

    df_compare['Cosine_ge_0.7'] = df_compare['Cosine_Score'] >= 0.7

    has_answer = df_compare['Text_answer'].astype(str).str.strip() != ""
    has_output = df_compare['Text_output'].astype(str).str.strip() != ""
    good_cos   = df_compare['Cosine_ge_0.7'] == True

    df_compare['Matrix'] = ""

    df_compare.loc[~has_answer & ~has_output, 'Matrix'] = "TN"
    df_compare.loc[has_answer & has_output & good_cos, 'Matrix'] = "TP"
    df_compare.loc[has_answer & ~has_output, 'Matrix'] = "FN"
    df_compare.loc[has_answer & has_output & ~good_cos, 'Matrix'] = "FN"
    df_compare.loc[~has_answer & has_output, 'Matrix'] = "FP"

    return df_compare


def compute_run_metrics(df_compare_filtered: pd.DataFrame):
    labels = df_compare_filtered['Matrix'].astype(str).str.strip()

    TP = (labels == 'TP').sum()
    TN = (labels == 'TN').sum()
    FP = (labels == 'FP').sum()
    FN = (labels == 'FN').sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom = TP + TN + FP + FN
    accuracy = (TP + TN) / denom if denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


def evaluate_one_model(model_dir: str, df_answer_cleaned: pd.DataFrame):
    model_name = os.path.basename(model_dir)
    model_output_root = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(model_output_root, exist_ok=True)

    results = []

    for shot in SHOT_FOLDERS:
        csv_dir = os.path.join(model_dir, shot, "csv")
        if not os.path.isdir(csv_dir):
            print(f"Skip: no folder -> {csv_dir}")
            continue

        shot_eval_dir = os.path.join(model_output_root, shot)
        os.makedirs(shot_eval_dir, exist_ok=True)

        for filename in sorted(os.listdir(csv_dir)):
            if not filename.endswith(".csv"):
                continue

            run_path = os.path.join(csv_dir, filename)
            print(f"--> Processing model={model_name}, shot={shot}, file={filename}")

            try:
                df_compare_filtered = clean_and_merge(run_path, df_answer_cleaned)
            except Exception as e:
                print(f"!!! ERROR in model={model_name}, shot={shot}, file={filename}")
                print(f"    {e}")
                continue

            run_name = os.path.splitext(filename)[0]
            out_name = f"{run_name}_cosine_eval.csv"
            out_path = os.path.join(shot_eval_dir, out_name)

            df_compare_filtered.to_csv(out_path, index=False)
            print(f"    Saved per-run eval to {out_path}")

            metrics = compute_run_metrics(df_compare_filtered)

            row = {
                "model": model_name,
                "shot": shot,
                "run": run_name,
                **metrics,
            }
            results.append(row)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_shot_avg = (
        df_results
        .groupby(["model", "shot"], as_index=False)[["precision", "recall", "f1", "accuracy"]]
        .mean()
    )

    # Save each model's CSV summaries
    df_results.to_csv(os.path.join(model_output_root, "cosine_eval_per_run.csv"), index=False)
    df_shot_avg.to_csv(os.path.join(model_output_root, "cosine_eval_per_shot.csv"), index=False)

    return df_results, df_shot_avg


def evaluate_all_models():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    df_answer_cleaned = load_answer_key(SRC_XLSX, SHEET)

    model_dirs = []
    for name in sorted(os.listdir(TASK1_ROOT)):
        full_path = os.path.join(TASK1_ROOT, name)
        if os.path.isdir(full_path) and name.startswith("Task1_") and name != "Task1_cosine_evaluation":
            model_dirs.append(full_path)

    if not model_dirs:
        print("No model folders found.")
        return

    all_per_run = []
    all_per_shot = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)

        df_per_run, df_per_shot = evaluate_one_model(model_dir, df_answer_cleaned)

        if df_per_run.empty:
            print(f"No results for {model_name}")
            continue

        all_per_run.append(df_per_run)
        all_per_shot.append(df_per_shot)

    if not all_per_run:
        print("No evaluation results found.")
        return

    df_all_per_run = pd.concat(all_per_run, ignore_index=True)
    df_all_per_shot = pd.concat(all_per_shot, ignore_index=True)

    df_all_per_run = df_all_per_run.sort_values(by=["model", "shot", "run"]).reset_index(drop=True)
    df_all_per_shot = df_all_per_shot.sort_values(by=["model", "shot"]).reset_index(drop=True)

    with pd.ExcelWriter(FINAL_EXCEL_PATH, engine="openpyxl") as writer:
        df_all_per_run.to_excel(writer, sheet_name="ALL_cosine_eval_per_run", index=False)
        df_all_per_shot.to_excel(writer, sheet_name="ALL_cosine_eval_per_shot", index=False)

    df_all_per_run.to_csv(os.path.join(OUTPUT_ROOT, "ALL_cosine_eval_per_run.csv"), index=False)
    df_all_per_shot.to_csv(os.path.join(OUTPUT_ROOT, "ALL_cosine_eval_per_shot.csv"), index=False)

    print(f"\nDone. Final Excel saved at:\n{FINAL_EXCEL_PATH}")


if __name__ == "__main__":
    evaluate_all_models()