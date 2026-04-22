import os, time, re, pandas as pd
import google.generativeai as genai

# ================== Gemini config ==================
MODEL_NAME  = "gemini-2.5-pro"
TEMPERATURE = 0
TOP_P       = 0.05

genai.configure(api_key="API_KEY_HERE")
model = genai.GenerativeModel(MODEL_NAME)

SRC_XLSX   = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
SHEET      = "Sheet2"
TEXT_COL   = "Selected Content"
TOPIC_COL  = "Topic"
POSNEG_COL = "Pos/Neg"
ID_COL     = "Column1"

TOPICS = [
    "room", "facility", "location", "staff", "food", "price",
    "check-in", "check-out", "taxi-issue", "booking-issue"
]

N_RUNS = 3

SAMPLE_IDS = []
LIMIT_ROWS = None

PRINT_TO_CONSOLE = True

CHECKPOINT_EVERY = 20
# ================== helpers ==================
def norm(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def topic_matches(row_topic: str, target: str) -> bool:
    return norm(row_topic) == norm(target)

def polarity(val: str):
    val = norm(val)
    if val in {"positive", "pos", "good", "p"}: return "good"
    if val in {"negative", "neg", "bad", "n"}:  return "bad"
    return None

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")

def coerce_id(x):
    try:
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, (int, str)):
            return x
        return int(x)
    except Exception:
        return x

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
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": TEMPERATURE, "top_p": TOP_P},
    )
    return (resp.text or "").strip()

def to_snake_phrase(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s_]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")

def postformat_aba(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.replace("\r\n", "\n").strip()
    lines = text.split("\n")
    tokens = []

    def clean_line(s: str) -> str:
        s = s.strip()
        low = s.lower()

        if not s:
            return ""
        if low.startswith("[raw") or low.startswith("here is") or low.startswith("based on"):
            return ""
        if "structured aba format" in low:
            return ""
        if low.startswith("argument "):
            return ""

        s = re.sub(r"^\s*[-*•]+\s*", "", s)
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)
        s = re.sub(r"^\s*(support|contrary)\s*:?\s*", "", s, flags=re.I)

        s = s.strip().strip('"').strip("'").strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for ln in lines:
        p = clean_line(ln)
        if not p:
            continue

        lower = p.lower()
        if lower.startswith("no evident not "):
            core = p[len("no evident not "):].strip()
            tok = "no_evident_not_" + to_snake_phrase(core)
        elif lower.startswith("have evident "):
            core = p[len("have evident "):].strip()
            tok = "have_evident_" + to_snake_phrase(core)
        else:
            tok = to_snake_phrase(p)

        if tok:
            tokens.append(tok)

    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return " , ".join(out)

def load_done_keys(run_csv_path: str, master_xlsx: str, test_col: str) -> set:
    """
    Returns set of (ID, Topic) keys that are already done for this run,
    reading from BOTH:
      - run CSV (append logs)
      - master Excel (so resume still works if CSV is missing)
    """
    done = set()

    if os.path.exists(run_csv_path):
        try:
            d = pd.read_csv(run_csv_path, usecols=["ID", "Topic"])
            d["ID"] = d["ID"].astype(str)
            d["Topic"] = d["Topic"].astype(str)
            done.update(zip(d["ID"], d["Topic"]))
        except Exception:
            pass

    if os.path.exists(master_xlsx):
        try:
            m = pd.read_excel(master_xlsx, usecols=["ID", "Topic", test_col])
            m["ID"] = m["ID"].astype(str)
            m["Topic"] = m["Topic"].astype(str)
            m[test_col] = m[test_col].astype("string").fillna("")
            m_done = m[m[test_col].str.strip() != ""]
            done.update(zip(m_done["ID"], m_done["Topic"]))
        except Exception:
            pass

    return done

def append_csv_rows(rows: list, path: str):
    if not rows:
        return
    df_part = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df_part.to_csv(path, index=False, mode="a", header=write_header)

def upsert_master_excel(master_xlsx: str, batch_rows: list, cols: list):
    """
    Upsert into master Excel on key (ID, Topic).
    - Updates ONLY the columns present in batch_rows (prevents wiping other Test columns).
    - Preserves existing Label if incoming Label is blank.
    """
    if not batch_rows:
        return

    upd = pd.DataFrame(batch_rows).copy()

    if os.path.exists(master_xlsx):
        master = pd.read_excel(master_xlsx)
    else:
        master = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in master.columns:
            master[c] = ""

    master["ID"] = master["ID"].astype(str)
    master["Topic"] = master["Topic"].astype(str)
    upd["ID"] = upd["ID"].astype(str)
    upd["Topic"] = upd["Topic"].astype(str)

    m = master.set_index(["ID", "Topic"])
    u = upd.set_index(["ID", "Topic"])

    upd_cols = [c for c in upd.columns if c not in ["ID", "Topic"]]
    common = m.index.intersection(u.index)

    if len(common) > 0:
        for c in upd_cols:
            if c == "Label":
                new = u.loc[common, c].astype("string").fillna("")
                old = m.loc[common, c].astype("string").fillna("")
                m.loc[common, c] = new.where(new.str.strip() != "", old)
            else:
                m.loc[common, c] = u.loc[common, c].astype("string").fillna("")

    new_idx = u.index.difference(m.index)
    if len(new_idx) > 0:
        new_rows = pd.DataFrame("", index=new_idx, columns=m.columns)
        for c in upd_cols:
            new_rows[c] = u.loc[new_idx, c].astype("string").fillna("")
        m = pd.concat([m, new_rows], axis=0)

    out = m.reset_index()[cols]
    out.to_excel(master_xlsx, index=False)

df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]

required = {TEXT_COL, TOPIC_COL, POSNEG_COL, ID_COL}
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in Excel: {missing}. Found: {list(df.columns)}")

def run_one_topic(TARGET_TOPIC: str):
    # select rows
    sel_rows = []
    SAMPLE_SET = set(SAMPLE_IDS) if SAMPLE_IDS else None

    for idx, r in df.iterrows():
        text   = r.get(TEXT_COL, "")
        topic  = r.get(TOPIC_COL, "")
        posneg = r.get(POSNEG_COL, "")
        rid    = coerce_id(r.get(ID_COL, None))

        if not isinstance(text, str) or not text.strip():
            continue
        if not topic_matches(topic, TARGET_TOPIC):
            continue

        pol = polarity(posneg)
        if pol is None:
            continue

        if rid in (None, "") or (isinstance(rid, float) and pd.isna(rid)):
            rid = idx + 1

        rid = str(rid)  # stable

        if SAMPLE_SET is not None and (rid not in set(map(str, SAMPLE_SET))):
            continue

        claim = f"{pol}_{snake(TARGET_TOPIC)}"
        sel_rows.append({"ID": rid, "Text": text.strip(), "Claim": claim})

    df_sel = pd.DataFrame(sel_rows).reset_index(drop=True)

    if (SAMPLE_SET is None) and isinstance(LIMIT_ROWS, int):
        df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

    if df_sel.empty:
        if PRINT_TO_CONSOLE:
            print(f"\n[SKIP] No matching rows for topic: {TARGET_TOPIC}")
        return

    topic_snake = snake(TARGET_TOPIC)
    out_dir = f"/Users/nonny/Downloads/Senior Project 2025/Task2/Gemini/Task2_gemini/ABA_task2_{topic_snake}_gemini"
    log_dir = os.path.join(out_dir, "log")
    csv_dir = os.path.join(out_dir, "CSV")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    cols = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, N_RUNS + 1)] + ["Label"]
    master_xlsx = os.path.join(out_dir, f"{topic_snake}_gemini_sheet.xlsx")
    
    if not os.path.exists(out_dir):
            raise SystemExit(f"[STOP] out_dir not found: {out_dir}")

    for run in range(1, N_RUNS + 1):
        run_csv_path = os.path.join(csv_dir, f"{topic_snake}_run{run}.csv")
        run_raw_path = os.path.join(log_dir, f"{topic_snake}_run{run}_raw.csv")
        test_col = f"Test {run}"

        if PRINT_TO_CONSOLE:
            print(f"\n========== TOPIC {TARGET_TOPIC} | RUN {run} / {N_RUNS} ==========")
            print("[DEBUG] out_dir     =", out_dir)
            print("[DEBUG] run_csv_path=", run_csv_path, "exists?", os.path.exists(run_csv_path))
            print("[DEBUG] master_xlsx =", master_xlsx,  "exists?", os.path.exists(master_xlsx))
            print("[DEBUG] csv_dir list =", os.listdir(csv_dir) if os.path.exists(csv_dir) else "NO csv_dir")

        done_keys = load_done_keys(run_csv_path, master_xlsx, test_col)
        

        if PRINT_TO_CONSOLE:
            print(f"[RESUME] Done rows for this run: {len(done_keys)}")

        batch_post, batch_raw = [], []
        new_count = 0

        for _, r in df_sel.iterrows():
            rid = str(r["ID"])
            key = (rid, r["Claim"])
            if key in done_keys:
                continue

            prm = build_prompt(r["Text"], r["Claim"])

            ans_raw, ans, err = "", "", ""
            try:
                ans_raw = ask_llm(prm)
                ans = postformat_aba(ans_raw)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                ans_raw = f"[ERROR] {err}"
                ans = f"[ERROR] {err}"

            post_row = {
                "ID": rid,
                "Prompt": prm,
                "Topic": r["Claim"],
                f"Test {run}": ans,
            }
            raw_row = {
                "ID": rid,
                "Prompt": prm,
                "Topic": r["Claim"],
                f"Raw {run}": ans_raw,
                "Error": err
            }

            batch_post.append(post_row)
            batch_raw.append(raw_row)

            if PRINT_TO_CONSOLE:
                print(f"[Run {run}] ID={rid}: {ans}")

            new_count += 1

            if new_count % CHECKPOINT_EVERY == 0:
                append_csv_rows(batch_post, run_csv_path)
                append_csv_rows(batch_raw, run_raw_path)

                excel_batch = []
                for br in batch_post:
                    excel_batch.append({
                        "ID": br["ID"],
                        "Prompt": br["Prompt"],
                        "Topic": br["Topic"],
                        f"Test {run}": br[f"Test {run}"],
                        "Label": "" 
                    })

                upsert_master_excel(master_xlsx, excel_batch, cols)

                batch_post.clear()
                batch_raw.clear()

            time.sleep(0.25)

        # flush remainder
        append_csv_rows(batch_post, run_csv_path)
        append_csv_rows(batch_raw, run_raw_path)

        excel_batch = []
        for br in batch_post:
            excel_batch.append({
                "ID": br["ID"],
                "Prompt": br["Prompt"],
                "Topic": br["Topic"],
                f"Test {run}": br[f"Test {run}"],
                "Label": ""
            })

        upsert_master_excel(master_xlsx, excel_batch, cols)

        batch_post.clear()
        batch_raw.clear()
        


    print(f"\nSaved raw logs -> {log_dir}")
    print(f"Saved postformatted CSVs -> {csv_dir}")
    print(f"Updated Excel -> {master_xlsx}")

# ================== run all topics ==================
for tp in TOPICS:
    run_one_topic(tp)