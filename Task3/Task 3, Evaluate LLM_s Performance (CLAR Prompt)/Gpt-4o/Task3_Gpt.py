import os, time, re
import pandas as pd
from datetime import datetime

from openai import OpenAI

MODEL_NAME  = "gpt-4o"

SAFE_TAG    = MODEL_NAME.replace(":", "-")
TEMPERATURE = 0.0
TOP_P       = 0.05

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it like:\n"
        "  export OPENAI_API_KEY='YOUR_KEY'\n"
        "or set it in your environment variables."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_DIR  = "/Users/nonny/Downloads/Senior Project 2025/Task3-Voted Dataset"

INPUT_FILES = [
    "1. Verify - Task 3 - Check-out (Silver).xlsx",
    "2. Verify - Task 3 - Check-in (Silver).xlsx",
    "3. Verify - Task 3 - Price (Silver).xlsx",
    "4. Verify - Task 3 - Staff (Silver).xlsx",
]

PHRASE_A_COL = "Assumption"      
PHRASE_B_COL = "Proposition"  
ID_COL       = ""           

SHEETS_TO_RUN = ["Contrary(P)Body(N)", "Contrary(N)Body(P)", "Contrary(P)Body(P)","Contrary(N)Body(N)"]

N_RUNS           = 3
BATCH_SIZE       = 20
SLEEP_SEC        = 0.25
RESUME           = True
PRINT_TO_CONSOLE = True

# 0-shot vs 1-shot
USE_ONE_SHOT = False
SHOT_SUFFIX = "_1shot" if USE_ONE_SHOT else "_0shot"
RUN_BASE = f"task3_contrary_{SAFE_TAG}{SHOT_SUFFIX}"

EXAMPLE_PHRASE_B = "no_one_answer_phone"
EXAMPLE_PHRASE_A = "no_evident_not_prompt_response_staff"
EXAMPLE_ANSWER   = "Yes"

def norm(x) -> str:
    return str(x).strip() if x is not None else ""

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None

def safe_file_token(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "sheet"

def file_tag_from_name(filename: str) -> str:
    low = filename.lower()
    if "check-out" in low or "checkout" in low:
        return "checkout"
    if "check-in" in low or "checkin" in low:
        return "checkin"
    if "price" in low:
        return "price"
    if "staff" in low:
        return "staff"
    return safe_file_token(os.path.splitext(filename)[0])[:40]

def make_key(rid: str, a: str, b: str) -> str:
    return f"{rid}||{a}||{b}"

def normalize_yes_no(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"\b(yes|no)\b", s, flags=re.IGNORECASE)
    if not m:
        return ""
    return "Yes" if m.group(1).lower() == "yes" else "No"

def get_prompt_0shot(phrase_a: str, phrase_b: str) -> str:
    return (
        "Is phrase B contrary to phrase A? Answer exactly 'Yes' or 'No' only.\n"
        "No explanations.\n\n"
        f"[Question] Phrase B is {phrase_b}. Phrase A is {phrase_a}."
    )

def get_prompt_1shot(phrase_a: str, phrase_b: str) -> str:
    return (
        "Is phrase B contrary to phrase A? Answer exactly 'Yes' or 'No' only.\n"
        "No explanations.\n\n"
        f"[Example] Phrase B is {EXAMPLE_PHRASE_B}. Phrase A is {EXAMPLE_PHRASE_A}. "
        f"Answer is '{EXAMPLE_ANSWER}'.\n\n"
        f"[Question] Phrase B is {phrase_b}. Phrase A is {phrase_a}."
    )

def build_prompt(phrase_a: str, phrase_b: str) -> str:
    return get_prompt_1shot(phrase_a, phrase_b) if USE_ONE_SHOT else get_prompt_0shot(phrase_a, phrase_b)

def ask_llm(prompt_text: str) -> str:
    """
    Returns the model raw text (or [ERROR]...).
    """
    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=prompt_text,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def enforce_run_csv_cols(df: pd.DataFrame, run: int) -> pd.DataFrame:
    cols = ["ID", "Prompt", "PhraseA", "PhraseB", f"Test {run}"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def enforce_log_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Timestamp", "Run", "ID", "PhraseA", "PhraseB", "Prompt", "RawOutput"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def load_done_keys(run_csv_path: str) -> set[str]:
    if not (RESUME and os.path.exists(run_csv_path)):
        return set()
    try:
        df_old = pd.read_csv(run_csv_path, dtype=str).fillna("")
        need = {"ID", "PhraseA", "PhraseB"}
        if not need.issubset(df_old.columns):
            return set()
        return set(make_key(rid, a, b) for rid, a, b in zip(df_old["ID"], df_old["PhraseA"], df_old["PhraseB"]))
    except Exception:
        return set()

def build_master_wide(df_input: pd.DataFrame, csv_dir: str, run_base: str) -> pd.DataFrame:
    wide = pd.DataFrame({
        "ID": df_input["ID"].astype(str),
        "Prompt": df_input["Prompt"].astype(str),
        "PhraseA": df_input["PhraseA"].astype(str),
        "PhraseB": df_input["PhraseB"].astype(str),
    })

    for run in range(1, N_RUNS + 1):
        run_csv = os.path.join(csv_dir, f"{run_base}_run{run}.csv")
        if not os.path.exists(run_csv):
            wide[f"Test {run}"] = ""
            continue
        rdf = pd.read_csv(run_csv, dtype=str).fillna("")
        rdf = enforce_run_csv_cols(rdf, run)
        rdf = rdf[["ID", "PhraseA", "PhraseB", f"Test {run}"]]
        wide = wide.merge(rdf, on=["ID", "PhraseA", "PhraseB"], how="left")

    cols = ["ID", "Prompt", "PhraseA", "PhraseB"] + [f"Test {i}" for i in range(1, N_RUNS + 1)]
    for c in cols:
        if c not in wide.columns:
            wide[c] = ""
    wide = wide[cols].drop_duplicates(subset=["ID", "PhraseA", "PhraseB"], keep="last")
    return wide

def resolve_sheet_name(actual_names: list[str], target: str) -> str | None:
    if target in actual_names:
        return target
    lookup = {s.strip(): s for s in actual_names}
    return lookup.get(target.strip())

def run_one_sheet(sheet_name: str, input_xlsx: str, out_root: str, run_base_prefix: str) -> pd.DataFrame:
    sheet_dir = os.path.join(out_root, sheet_name)
    csv_dir   = os.path.join(sheet_dir, "csv")
    log_dir   = os.path.join(sheet_dir, "log")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    sheet_token = safe_file_token(sheet_name)
    run_base = f"{run_base_prefix}_{sheet_token}"

    master_xlsx = os.path.join(out_root, f"{run_base}_sheet.xlsx")

    df = pd.read_excel(input_xlsx, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    phrase_b_col = PHRASE_B_COL.strip()
    if not phrase_b_col:
        phrase_b_col = detect_column(df, ["Proposition", "B", "Phrase B", "PhraseB", "OriginalB"])
    if not phrase_b_col:
        raise ValueError(f"[{sheet_name}] Cannot find Phrase B column. Set PHRASE_B_COL.")

    if PHRASE_A_COL not in df.columns:
        raise ValueError(f"[{sheet_name}] Missing Phrase A column '{PHRASE_A_COL}'.")

    id_col = ID_COL.strip()
    if not id_col:
        id_col = detect_column(df, ["ID", "Id", "id", "Column1", "Index"])
    use_row_index_as_id = (id_col is None)

    rows = []
    for idx, r in df.iterrows():
        a = norm(r.get(PHRASE_A_COL, ""))
        b = norm(r.get(phrase_b_col, ""))
        if not a or not b:
            continue
        rid = str(idx + 1) if use_row_index_as_id else norm(r.get(id_col, idx + 1))
        prompt = build_prompt(a, b)
        rows.append({"ID": rid, "PhraseA": a, "PhraseB": b, "Prompt": prompt})

    df_input = pd.DataFrame(rows).reset_index(drop=True)
    if df_input.empty:
        empty = pd.DataFrame(columns=["ID","Prompt","PhraseA","PhraseB","Test 1","Test 2","Test 3"])
        empty.to_excel(master_xlsx, index=False)
        return empty

    for run in range(1, N_RUNS + 1):
        run_csv = os.path.join(csv_dir, f"{run_base}_run{run}.csv")
        run_log = os.path.join(log_dir, f"{run_base}_run{run}.csv")

        done_keys = load_done_keys(run_csv)

        if PRINT_TO_CONSOLE:
            print(f"\n[{sheet_name}] ===== RUN {run}/{N_RUNS} =====")
            if RESUME and done_keys:
                print(f"[{sheet_name}] resume done: {len(done_keys)} rows")

        results_exists = os.path.exists(run_csv)
        log_exists = os.path.exists(run_log)
        batch_rows = []

        for _, rr in df_input.iterrows():
            rid = str(rr["ID"])
            a = rr["PhraseA"]
            b = rr["PhraseB"]
            prompt = rr["Prompt"]

            key = make_key(rid, a, b)
            if RESUME and key in done_keys:
                continue

            raw = ask_llm(prompt)
            yn = normalize_yes_no(raw)
            test_value = yn if yn else raw

            log_row = {
                "Timestamp": now_ts(),
                "Run": run,
                "ID": rid,
                "PhraseA": a,
                "PhraseB": b,
                "Prompt": prompt,
                "RawOutput": raw
            }
            df_log = enforce_log_cols(pd.DataFrame([log_row]))
            df_log.to_csv(run_log, index=False, mode=("a" if log_exists else "w"), header=(not log_exists))
            log_exists = True

            batch_rows.append({
                "ID": rid,
                "Prompt": prompt,
                "PhraseA": a,
                "PhraseB": b,
                f"Test {run}": test_value
            })
            done_keys.add(key)

            if PRINT_TO_CONSOLE:
                print(f"[{sheet_name}] [Run {run}] ID={rid} => {test_value}")

            if len(batch_rows) >= BATCH_SIZE:
                df_part = enforce_run_csv_cols(pd.DataFrame(batch_rows), run)
                df_part.to_csv(run_csv, index=False, mode=("a" if results_exists else "w"), header=(not results_exists))
                results_exists = True
                batch_rows.clear()

                wide = build_master_wide(df_input, csv_dir, run_base)
                wide.to_excel(master_xlsx, index=False)

            time.sleep(SLEEP_SEC)

        if batch_rows:
            df_part = enforce_run_csv_cols(pd.DataFrame(batch_rows), run)
            df_part.to_csv(run_csv, index=False, mode=("a" if results_exists else "w"), header=(not results_exists))
            batch_rows.clear()

        try:
            df_full = pd.read_csv(run_csv, dtype=str).fillna("")
            df_full = enforce_run_csv_cols(df_full, run)
            df_full = df_full.drop_duplicates(subset=["ID","PhraseA","PhraseB"], keep="last")
            df_full.to_csv(run_csv, index=False)
        except Exception:
            pass

        wide = build_master_wide(df_input, csv_dir, run_base)
        wide.to_excel(master_xlsx, index=False)

    final_wide = build_master_wide(df_input, csv_dir, run_base)
    final_wide.to_excel(master_xlsx, index=False)
    return final_wide

def run_one_workbook(input_xlsx: str, out_root: str):
    os.makedirs(out_root, exist_ok=True)
    all_sheets_xlsx = os.path.join(out_root, f"{RUN_BASE}_ALL_sheets.xlsx")

    xls = pd.ExcelFile(input_xlsx)
    sheet_names = xls.sheet_names

    print("\n[INFO] Workbook:", input_xlsx)
    print("[INFO] Sheets found:")
    for s in sheet_names:
        print(" -", repr(s))

    targets_raw = SHEETS_TO_RUN if SHEETS_TO_RUN else sheet_names
    targets = []
    for t in targets_raw:
        real = resolve_sheet_name(sheet_names, t)
        if real is None:
            print(f"[WARN] Sheet not found (skip): {repr(t)}")
        else:
            targets.append(real)

    if not targets:
        print("[WARN] No valid sheets to run for this workbook.")
        return

    all_outputs = {}
    failures = []

    for s in targets:
        try:
            print(f"\n[RUN] Sheet: {s}")
            all_outputs[s] = run_one_sheet(s, input_xlsx, out_root, RUN_BASE)
        except Exception as e:
            failures.append((s, str(e)))
            print(f"[ERROR] Sheet failed: {s} -> {e}")
            continue

    if all_outputs:
        with pd.ExcelWriter(all_sheets_xlsx, engine="openpyxl") as writer:
            for s, wide_df in all_outputs.items():
                wide_df.to_excel(writer, sheet_name=s[:31], index=False)
        print(f"\n[OK] Combined workbook saved: {all_sheets_xlsx}")
    else:
        print("\n[WARN] No sheets succeeded, so ALL_sheets.xlsx was not created.")

    if failures:
        print("\nSome sheets failed:")
        for s, err in failures:
            print(f" - {s}: {err}")

def main():
    for fname in INPUT_FILES:
        input_xlsx = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(input_xlsx):
            print(f"[SKIP] Not found: {input_xlsx}")
            continue

        tag = file_tag_from_name(fname)
        out_root = os.path.join(INPUT_DIR, f"{RUN_BASE}_{tag}")

        print("\n" + "="*80)
        print(f"[FILE] {fname}")
        print(f"[OUT ] {out_root}")
        print("="*80)

        run_one_workbook(input_xlsx, out_root)

    print("\nDONE All workbooks finished.")

if __name__ == "__main__":
    main()