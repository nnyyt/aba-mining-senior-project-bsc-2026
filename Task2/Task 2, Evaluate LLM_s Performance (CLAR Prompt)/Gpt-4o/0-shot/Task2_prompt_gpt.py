import os, time, re, pandas as pd
import openai

# ================== OpenAI client ==================
client = openai.OpenAI(api_key='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') 

# ================== Minimal config ==================
SRC_XLSX   = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (25-07-2025 Nonny Version).xlsx"
SHEET      = "Nonny Version"
TEXT_COL   = "Selected Content"
TOPIC_COL  = "Topic"
POSNEG_COL = "Pos/Neg"       # Positive / Negative (as in your sheet)
ID_COL     = "Column1"       # your ID column

TARGET_TOPIC = "room"       # run one topic at a time
N_RUNS       = 3             # e.g., 3 or 5

# Choose rows:
SAMPLE_IDS = []          
LIMIT_ROWS = None            # or set an int to take first N rows after filtering

MODEL_NAME  = "gpt-4o"
TEMPERATURE = 0
TOP_P       = 0.05

# Console printing
PRINT_TO_CONSOLE   = True

# ================== helpers ==================
def norm(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def topic_matches(row_topic: str, target: str) -> bool:
    return norm(row_topic) == norm(target)

def polarity(v: str):
    v = norm(v)
    if v in {"positive","pos","good","p"}: return "good"
    if v in {"negative","neg","bad","n"}:  return "bad"
    return None

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")

def coerce_id(v):
    try:
        # make 1.0 -> 1 for cleaner printing/files
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, (int, str)):
            return v
        return int(v)
    except Exception:
        return v

def contrary_prefix_for_claim(claim: str) -> str:
    """
    Map claim polarity to contrary prefix for prompt rule #5.
    good_* -> "no evident not"
    bad_*  -> "have evident"
    (fallback to 'no evident not' if format is unexpected)
    """
    c = norm(claim or "")
    if c.startswith("good_"):
        return "no evident not"
    if c.startswith("bad_"):
        return "have evident"
    return "no evident not"

def prompt_header(claim: str) -> str:
    contr = contrary_prefix_for_claim(claim)  # <-- dynamic
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
    return client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content.strip()

def to_snake_phrase(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s_]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")

def postformat_aba(raw: str) -> str:
    """
    Convert any sentence/numbered 'Support/Contrary' lines into
    ONE-LINE comma-separated snake_case tokens, with contraries
    prefixed 'no_evident_not_'.
    """
    if not isinstance(raw, str):
        raw = str(raw)
    text = raw.strip()

    # split by newline or comma
    parts = re.split(r"[\n,]+", text)
    tokens = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # remove numbering and labels
        p = re.sub(r"^\d+\.\s*", "", p, flags=re.I)
        p = re.sub(r"^(support|contrary)\s*:?\s*", "", p, flags=re.I)

        lower = p.lower()
        if lower.startswith("no evident not "):
            core = p[15:].strip()
            token = "no_evident_not_" + to_snake_phrase(core)
        else:
            token = to_snake_phrase(p)
        if token:
            tokens.append(token)

    # dedupe keep order
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return " , ".join(out)

# ================== load & select ==================
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

    claim = f"{pol}_{snake(TARGET_TOPIC)}"   # e.g., good_staff
    sel_rows.append({"ID": rid, "Text": text.strip(), "Claim": claim})

df_sel = pd.DataFrame(sel_rows).reset_index(drop=True)

if (SAMPLE_SET is None) and isinstance(LIMIT_ROWS, int):
    df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

if df_sel.empty:
    raise SystemExit("No matching rows after filtering/sampling.")

# ================== run N times ==================
topic_snake = snake(TARGET_TOPIC)
out_dir = f"ABA_task2_{topic_snake}_gpt_4o"
os.makedirs(out_dir, exist_ok=True)

all_runs_tests = []

for run in range(1, N_RUNS + 1):
    out_rows = []
    if PRINT_TO_CONSOLE:
        print(f"\n========== RUN {run} / {N_RUNS} ==========")
    for _, r in df_sel.iterrows():
        prm = build_prompt(r["Text"], r["Claim"])
        try:
            ans_raw = ask_llm(prm)
            ans = postformat_aba(ans_raw)  # format to your desired one-line list
        except Exception as e:
            ans = f"[ERROR] {type(e).__name__}: {e}"

        out_rows.append({
            "ID": r["ID"],
            "Prompt": prm,
            "Topic": r["Claim"],          # claim like good_staff
            f"Test {run}": ans
        })

        if PRINT_TO_CONSOLE:
            print(f"[Run {run}] ID={r['ID']}: {ans}")

        time.sleep(0.25)

    run_df = pd.DataFrame(out_rows)
    run_df.to_csv(os.path.join(out_dir, f"{topic_snake}_run{run}.csv"), index=False)
    all_runs_tests.append(run_df[[f"Test {run}"]])

# ================== build wide & append to Excel ==================
wide = pd.DataFrame({
    "ID":     df_sel["ID"],
    "Prompt": [build_prompt(t, c) for t, c in zip(df_sel["Text"], df_sel["Claim"])],
    "Topic":  df_sel["Claim"],
})
for run in range(1, N_RUNS + 1):
    wide = pd.concat([wide, all_runs_tests[run-1]], axis=1)

wide["Label"] = ""  # keep empty; or: wide["Label"] = wide["Test 1"]
cols = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, N_RUNS + 1)] + ["Label"]
wide = wide[cols]

master_xlsx = os.path.join(out_dir, f"{topic_snake}_gpt_4o_sheet.xlsx")
if os.path.exists(master_xlsx):
    prev = pd.read_excel(master_xlsx)
    for c in cols:
        if c not in prev.columns: prev[c] = ""
    for c in prev.columns:
        if c not in wide.columns: wide[c] = ""
    combined = pd.concat([prev[cols], wide[cols]], ignore_index=True)
    combined = combined.drop_duplicates(subset=cols, keep="last")
    combined.to_excel(master_xlsx, index=False)
else:
    wide.to_excel(master_xlsx, index=False)

print(f"\nSaved per-run CSVs in {out_dir} and appended to Excel: {master_xlsx}")
