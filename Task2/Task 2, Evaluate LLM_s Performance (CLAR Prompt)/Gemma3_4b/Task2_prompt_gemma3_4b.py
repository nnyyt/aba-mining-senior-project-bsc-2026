import os, time, re
import pandas as pd
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_TAG   = "gemma3:4b"        
SAFE_TAG    = MODEL_TAG.replace(":", "-")

SRC_XLSX   = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
SHEET      = "Sheet2"
TEXT_COL   = "Selected Content"
TOPIC_COL  = "Topic"
POSNEG_COL = "Pos/Neg"
ID_COL     = "Column1"

TARGET_TOPIC = "location"
N_RUNS       = 3

RESUME = True
CHUNK_SAVE_EVERY = 20  

SAMPLE_IDS = []     
LIMIT_ROWS = None    

TEMPERATURE = 0.0
TOP_P       = 0.05

PRINT_TO_CONSOLE = True
USE_POSTFORMAT = True

def norm(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def topic_matches(row_topic: str, target: str) -> bool:
    return norm(row_topic) == norm(target)

def polarity(v: str):
    v = norm(v)
    if v in {"positive", "pos", "good", "p"}:
        return "good"
    if v in {"negative", "neg", "bad", "n"}:
        return "bad"
    return None

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")

def coerce_id(v):
    try:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, (int, str)):
            return v
        return int(v)
    except Exception:
        return v

def make_key(rid, claim) -> str:
    return f"{str(rid).strip()}||{norm(claim)}"

def contrary_prefix_for_claim(claim: str) -> str:
    c = norm(claim or "")
    if c.startswith("good_"):
        return "no evident not"
    if c.startswith("bad_"):
        return "have evident"
    return "no evident not"

def prompt_header(claim: str) -> str:
    contr = contrary_prefix_for_claim(claim)
    return f"""Generate text in Assumption Based Argumentation (ABA) format from the given text. Use the following conditions carefully.

1. Claim is "{claim}".
2. Supports are written in short words with no adjectives and no adverbs if it is not necessary for understanding.
3. For each support, use vocab from the original text. Do not provide synonyms. Check grammar. Do not provide further opinion.
4. Add a contrary for each support as a new support.
5. Each contrary must use the same word as presented in the support and each contrary starts with "{contr}".
6. Regarding the format of answer, provide a list of all supports and contraries. Do not separate supports and contraries into separated sections. Do not provide assumptions. Do not provide claims.

[Text]
"""

def build_prompt(text: str, claim: str) -> str:
    return prompt_header(claim) + (text or "").strip()

def ask_llm(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_TAG,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}).get("content") or "").strip()
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"
    
def enforce_run_cols(df, run):
    col_test = f"Test {run}"
    cols = ["ID", "Prompt", "Topic", col_test]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def postformat_aba(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw or "")

    out = []
    label_head = re.compile(r'(?i)^\s*(supports?|contraries|support|contrary)\s*[:\-–—]\s*')

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fragments = re.split(r'\.\s+', line)
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue
            frag = label_head.sub('', frag)
            if re.match(r'(?i)^\s*(supports?|contraries)\s*:?\s*$', frag):
                continue
            frag = re.sub(r'[.,]\s*$', '', frag)
            token = re.sub(r'\s+', '_', frag.lower())
            if token:
                out.append(token)

    return " , ".join(out)

def _pairs_from_log_csv(path: str):
    if not os.path.exists(path):
        return set()
    try:
        df_old = pd.read_csv(path, dtype=str)
        if {"ID", "Claim"}.issubset(df_old.columns):
            tmp = df_old[["ID", "Claim"]].fillna("")
            return set((tmp["ID"].astype(str).str.strip() + "||" + tmp["Claim"].str.lower()).tolist())
        return set()
    except Exception:
        return set()

def save_master_excel(df_sel, out_dir, topic_snake, safe_tag, n_runs):
    master_xlsx = os.path.join(out_dir, f"{topic_snake}_{safe_tag}_sheet.xlsx")

    wide = pd.DataFrame({
        "ID":     df_sel["ID"].astype(str),
        "Prompt": [build_prompt(t, c) for t, c in zip(df_sel["Text"], df_sel["Claim"])],
        "Topic":  df_sel["Claim"].astype(str),
    })

    for i in range(1, n_runs + 1):
        run_csv_path = os.path.join(out_dir, f"{topic_snake}_run{i}.csv")
        if os.path.exists(run_csv_path):
            try:
                rdf = pd.read_csv(run_csv_path, dtype=str)
                need = ["ID", "Topic", f"Test {i}"]
                if all(c in rdf.columns for c in need):
                    rdf = rdf[need]
                    rdf["ID"] = rdf["ID"].astype(str)
                    rdf["Topic"] = rdf["Topic"].astype(str)
                    wide = wide.merge(rdf, on=["ID", "Topic"], how="left")
            except Exception as e:
                print(f"[Excel] Skip merging run {i} ({e})")

    if "Label" not in wide.columns:
        wide["Label"] = ""
        
    if os.path.exists(master_xlsx):
        try:
            prev = pd.read_excel(master_xlsx, dtype=str)
            for c in prev.columns:
                if c not in wide.columns:
                    wide[c] = prev[c]
        except Exception as e:
            print(f"[Excel] Warning: cannot read existing master ({e})")

    cols = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, n_runs + 1)] + ["Label"]
    for c in cols:
        if c not in wide.columns:
            wide[c] = ""

    wide = wide[cols]
    wide = wide.drop_duplicates(subset=["ID", "Topic"], keep="last")

    wide.to_excel(master_xlsx, index=False)
    if PRINT_TO_CONSOLE:
        print(f"[Excel] Updated: {master_xlsx}")
        
df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]

required = {TEXT_COL, TOPIC_COL, POSNEG_COL, ID_COL}
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in Excel: {missing}. Found: {list(df.columns)}")

sel_rows = []
SAMPLE_SET = set(SAMPLE_IDS) if SAMPLE_IDS else None

for idx, r in df.iterrows():
    text    = r.get(TEXT_COL, "")
    topic   = r.get(TOPIC_COL, "")
    posneg  = r.get(POSNEG_COL, "")
    rid     = coerce_id(r.get(ID_COL, None))

    if not isinstance(text, str) or not text.strip():
        continue
    if not topic_matches(topic, TARGET_TOPIC):
        continue

    pol = polarity(posneg)
    if pol is None:
        continue

    if rid in (None, "") or (isinstance(rid, float) and pd.isna(rid)):
        rid = idx + 1

    if SAMPLE_SET is not None and (rid not in SAMPLE_SET):
        continue

    claim = f"{pol}_{snake(TARGET_TOPIC)}"   # good_room / bad_room
    sel_rows.append({"ID": rid, "Text": text.strip(), "Claim": claim})

df_sel = pd.DataFrame(sel_rows).reset_index(drop=True)

if (SAMPLE_SET is None) and isinstance(LIMIT_ROWS, int):
    df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

if df_sel.empty:
    raise SystemExit("No matching rows after filtering/sampling.")

topic_snake = snake(TARGET_TOPIC)
out_dir = f"/Users/nonny/Downloads/Senior Project 2025/Task2/Gemma3/Task2_0shot/ABA_task2_{topic_snake}_{SAFE_TAG}_0shot"
os.makedirs(out_dir, exist_ok=True)

log_dir = os.path.join(out_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

for run in range(1, N_RUNS + 1):
    if PRINT_TO_CONSOLE:
        print(f"\n========== RUN {run} / {N_RUNS} ==========")

    run_csv_path = os.path.join(out_dir, f"{topic_snake}_run{run}.csv")
    log_csv_path = os.path.join(log_dir, f"log_run_{run}.csv")

    col_test = f"Test {run}"

    done_pairs = set()

    if RESUME and os.path.exists(run_csv_path):
        try:
            prev_run_df = pd.read_csv(run_csv_path, dtype=str)
            if {"ID", "Topic", col_test}.issubset(prev_run_df.columns):
                ok = (~prev_run_df[col_test].isna()) & (prev_run_df[col_test].astype(str).str.strip() != "")
                tmp = prev_run_df.loc[ok, ["ID", "Topic"]].fillna("")
                done_pairs |= set((tmp["ID"].astype(str).str.strip() + "||" + tmp["Topic"].str.lower()).tolist())
        except Exception as e:
            print(f"[Resume] Could not read existing CSV ({run_csv_path}): {e}")

    if RESUME:
        done_pairs |= _pairs_from_log_csv(log_csv_path)

    if PRINT_TO_CONSOLE and done_pairs:
        print(f"[Resume] Found {len(done_pairs)} completed (ID+Claim) pairs for run {run}.")

    existing_df = None
    if RESUME and os.path.exists(run_csv_path):
        try:
            existing_df = pd.read_csv(run_csv_path, dtype=str)
        except Exception as e:
            print(f"[Resume] Warning: cannot load existing run CSV: {e}")

    log_exists = os.path.exists(log_csv_path)

    out_rows = []
    processed_since_save = 0
    
    for _, r in df_sel.iterrows():
        rid = r["ID"]
        claim = r["Claim"]
        k = make_key(rid, claim)

        if k in done_pairs:
            if PRINT_TO_CONSOLE:
                print(f"[Run {run}] ID={rid} Claim={claim}: skipped (already done)")
            continue

        prm = build_prompt(r["Text"], claim)
        ans_raw = ask_llm(prm)
        ans = postformat_aba(ans_raw) if USE_POSTFORMAT else ans_raw

        pd.DataFrame([{
            "ID": str(rid).strip(),
            "Claim": claim,
            "Output": ans_raw
        }]).to_csv(
            log_csv_path,
            index=False,
            mode=("a" if log_exists else "w"),
            header=(not log_exists)
        )
        log_exists = True
        
        out_rows.append({
            "ID": str(rid).strip(),
            "Prompt": prm,
            "Topic": claim,
            col_test: ans
        })

        done_pairs.add(k)

        if PRINT_TO_CONSOLE:
            print(f"[Run {run}] ID={rid} Claim={claim}: {ans}")

        processed_since_save += 1

        if CHUNK_SAVE_EVERY and processed_since_save >= CHUNK_SAVE_EVERY:
            tmp_df = pd.DataFrame(out_rows)

            merged = pd.concat([existing_df, tmp_df], ignore_index=True) if existing_df is not None else tmp_df
            merged = merged.drop_duplicates(subset=["ID", "Topic"], keep="last")
            merged = merged[["ID", "Prompt", "Topic", col_test]]
            merged = enforce_run_cols(merged, run) 
            merged.to_csv(run_csv_path, index=False)

            if PRINT_TO_CONSOLE:
                print(f"[Run {run}] Partial save: {len(merged)} rows → {run_csv_path}")

            existing_df = merged
            out_rows = []
            processed_since_save = 0

            save_master_excel(df_sel, out_dir, topic_snake, SAFE_TAG, N_RUNS)

        time.sleep(0.25)
        
    run_df = pd.DataFrame(out_rows)

    if existing_df is not None and not run_df.empty:
        final_merged = pd.concat([existing_df, run_df], ignore_index=True)
    elif existing_df is not None:
        final_merged = existing_df
    else:
        final_merged = run_df

    if final_merged is None or final_merged.empty:
        pd.DataFrame(columns=["ID", "Prompt", "Topic", col_test]).to_csv(run_csv_path, index=False)
    else:
        need_cols = ["ID", "Prompt", "Topic", f"Test {run}"]
        for c in need_cols:
            if c not in final_merged.columns:
                final_merged[c] = ""

        final_merged = final_merged[need_cols]
        final_merged = final_merged.drop_duplicates(subset=["ID", "Topic"], keep="last")

        try:
            final_merged["_IDNUM"] = pd.to_numeric(final_merged["ID"], errors="coerce")
            final_merged = final_merged.sort_values(["_IDNUM", "Topic"]).drop(columns=["_IDNUM"])
        except Exception:
            final_merged = final_merged.sort_values(["ID", "Topic"])

        final_merged.to_csv(run_csv_path, index=False)

        if PRINT_TO_CONSOLE:
            print(f"[Run {run}] Saved {len(final_merged)} rows → {run_csv_path}")

    save_master_excel(df_sel, out_dir, topic_snake, SAFE_TAG, N_RUNS)

print("\nAll done.")