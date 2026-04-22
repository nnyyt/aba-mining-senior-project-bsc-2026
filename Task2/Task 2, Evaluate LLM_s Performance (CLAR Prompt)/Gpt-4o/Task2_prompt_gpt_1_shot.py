import os, time, re, pandas as pd
import openai

client = openai.OpenAI(api_key='API KEY HERE') 

SRC_XLSX   = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
SHEET      = "Sheet2"
TEXT_COL   = "Selected Content"
TOPIC_COL  = "Topic"
POSNEG_COL = "Pos/Neg"
ID_COL     = "Column1"

TARGET_TOPIC = "location"  
SAMPLE_IDS   = []   
LIMIT_ROWS   = None
N_RUNS       = 3

MODEL_NAME   = "gpt-4o"
TEMPERATURE  = 0
TOP_P        = 0.05
PRINT_TO_CONSOLE = True
SLEEP_SEC    = 0.25

USE_ONE_SHOT   = True
USE_POSTFORMAT = False
PRESERVE_DOTS  = False

ONE_SHOT_EXAMPLES = {
    "good_staff": {
        "text": "The staff were exceptional. So helpful and friendly. Went out of their way for us.",
        "output": "exceptional_staff , helpful_staff , friendly_staff , staff_go_out_of_their_way_for_us , no_evident_not_exceptional_staff , no_evident_not_helpful_staff , no_evident_not_friendly_staff , no_evident_not_staff_go_out_of_their_way_for_us"
    },
    "bad_staff": {
        "text": "The receptionist didn't speak English properly",
        "output": "staff_didn_t_speak_english_properly , have_evident_staff_didn_t_speak_english_properly"
    },
    "good_room": {
        "text": "The room was spacious and comfortable. Very clean with a nice view.",
        "output": "clean_room , well_decorated_room , modern_room , no_evident_not_clean_room , no_evident_not_well_decorated_room , no_evident_not_modern_room"
    },
    "bad_room": {
        "text": "The floor is not cleaned enough, full of dust, water does not drain and floods the room",
        "output": "not_clean_enough_floor , dust_on_floor , water_does_not_drain , floods_room , have_evident_not_clean_enough_floor , have_evident_dust_on_floor , have_evident_water_does_not_drain , have_evident_floods_room"
    },
    "good_facility": {
        "text": "The Hotel itself was very nice, spacious, clean and modern.",
        "output": "nice_hotel , spacious_hotel , clean_hotel , modern_hotel , no_evident_not_nice_hotel , no_evident_not_spacious_hotel , no_evident_not_clean_hotel , no_evident_not_modern_hotel"
    },
    "bad_facility": {
        "text": "when I came but the hotel doesn't have it's own parking lot so sometimes it might be difficult to find a parking spot.",
        "output": "no_own_parking_lot , difficult_to_find_parking , have_evident_no_own_parking_lot , have_evident_difficult_to_find_parking"
    },
    "good_location": {
        "text": "close to the airport, to very clean beach.",
        "output": "close_to_airport , close_to_clean_beach , no_evident_not_close_to_airport , no_evident_not_close_to_clean_beach"
    },
    "bad_location": {
        "text": "we were surprised at the location in a small side street,",
        "output": "located_in_small_side_street , have_evident_located_in_small_side_street"
    },
    "good_food": {
        "text": "Tasty food on the first floor, comfortable restaurant for both cozy evenings and calm work to escape the heat in the midday",
        "output": "tasty_food , comfortable_restaurant_for_cozy_evenings , comfortable_restaurant_for_calm_work , no_evident_not_tasty_food , no_evident_not_comfortable_restaurant_for_cozy_evenings , no_evident_not_comfortable_restaurant_for_calm_work"
    },
    "bad_food": {
        "text": "Ordered pizza one night and was not great, too stodgy",
        "output": "not_great_pizza , stodgy_pizza , have_evident_not_great_pizza , have_evident_stodgy_pizza"
    },
    "good_price": {
        "text": "Great value for money",
        "output": "great_value_for_money , no_evident_not_great_value_for_money"
    },
    "bad_price": {
        "text": "For what you get it was a bit expensive.",
        "output": "expensive_for_what_you_get , have_evident_expensive_for_what_you_get"
    },
    "good_check_in": {
        "text": "clear instructions left access to our room made easy",
        "output": "clear_instructions_to_access_room , no_evident_not_clear_instructions_to_access_room"
    },
    "bad_check_in": {
        "text": "We had a late check in and there was no one in the hotel to give us keys and the door was locked.",
        "output": "no_one_gave_keys_on_late_check_in , door_locked_on_late_check_in , have_evident_no_one_gave_keys_on_late_check_in , have_evident_door_locked_on_late_check_in"
    },
    "good_check_out": {
        "text": "super fast checkin/checkout",
        "output": "fast_check_out , no_evident_not_fast_check_out"
    },
    "bad_check_out": {
        "text": "4. the hotel did not allow to turn over a couple of hours before the delayed plane;",
        "output": "staff_did_not_allow_late_check_out_for_delayed_plane , have_evident_staff_did_not_allow_late_check_out_for_delayed_plane"
    },
    "good_taxi_issue": {
        "text": "We arrived at 2:30 am at the airport, a shuttle taxi to the hotel was waiting for us.",
        "output": "taxi_waiting_for_us , no_evident_not_taxi_waiting_for_us"
    },
    "bad_taxi_issue": {
        "text": "Taxi drivers who cooperate with hotel, both times, from airport and back, charged 2-3 EUR above the price agreed with the hotel (20 EUR).",
        "output": "charged_above_agreed_price , no_alternative_transport_at_midnight , have_evident_charged_above_agreed_price , have_evident_no_alternative_transport_at_midnight"
    },
    "bad_booking_issue": {
        "text": "However when I booked this property in advance it was a higher price but booking.com reduced the price before my travel date and the hotel refused to accept the lower price which I felt was unfair for me as they charged twice for cancellation and rebooking. Maybe this was a booking.com issue or the hotel but I was quite unhappy about this since it was not justified by the owner. I spoke to Ali from booking.com and he said there was no cancellation charge but they charged me",
        "output": "mismatch_cancellation_charge_policy , price_reduced_after_booking , hotel_refused_to_accept_lower_price , have_evident_mismatch_cancellation_charge_policy , have_evident_price_reduced_after_booking , have_evident_hotel_refused_to_accept_lower_price"
    },
}

def norm(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")

def topic_matches(row_topic: str, target: str) -> bool:
    return norm(row_topic) == norm(target)

def polarity(v: str):
    v = norm(v)
    if v in {"positive","pos","good","p"}: return "good"
    if v in {"negative","neg","bad","n"}:  return "bad"
    return None

def coerce_id(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()
    except Exception:
        return None

def contrary_prefix_for_claim(claim: str) -> str:
    c = norm(claim or "")
    if c.startswith("good_"): return "no evident not"
    if c.startswith("bad_"):  return "have evident"
    return "no evident not"

def get_example_for_claim(claim: str):
    ex = ONE_SHOT_EXAMPLES.get(norm(claim))
    if not ex:
        return None, None
    return ex["text"], ex["output"]

def get_system_prompt(claim: str) -> str:
    contr = contrary_prefix_for_claim(claim)
    return (
        "Generate text in Assumption Based Argumentation (ABA) format from the given text. Use the following conditions carefully.\n"
        f"1. Claim is \"{claim}\".\n"
        "2. Supports are written in short words with no adjectives and no adverbs if it is not necessary for understanding.\n"
        "3. For each support, use vocab from the original text. Do not provide synonyms. Check grammar. Do not provide further opinion.\n"
        "4. Add a contrary for each support as a new support.\n"
        f"5. Each contrary must use the same word as presented in the support and each contrary starts with \"{contr}\".\n"
        "6. Regarding the format of answer, provide a list of all supports and contraries. Do not separate supports and contraries into separated sections. Do not provide assumptions. Do not provide claims."
    )

def build_messages(text: str, claim: str) -> list:
    msgs = [{"role": "system", "content": get_system_prompt(claim)}]
    if USE_ONE_SHOT:
        ex_text, ex_out = get_example_for_claim(claim)
        if ex_text and ex_out:
            msgs.append({"role": "user", "content": "[Text] " + ex_text})
            msgs.append({"role": "assistant", "content": ex_out})
    msgs.append({"role": "user", "content": "[Text] " + (text or "").strip()})
    return msgs

def stringify_messages(msgs: list) -> str:
    role_map = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
    blocks = []
    for m in msgs:
        role = role_map.get(m.get("role","").lower(), m.get("role","").upper())
        content = (m.get("content") or "").strip()
        blocks.append(f"{role}:\n{content}")
    return "\n\n---\n\n".join(blocks)

def ask_llm_messages(messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()

def to_snake_phrase(s: str) -> str:
    s = (s or "").lower()
    if PRESERVE_DOTS:
        s = re.sub(r"[^a-z0-9._\s]", " ", s)
    else:
        s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def postformat_aba(raw: str, claim: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw or "")
    text = raw.strip()
    if not text:
        return ""

    expected = contrary_prefix_for_claim(claim).lower()
    expected_snake = expected.replace(" ", "_")
    alt = "have evident" if expected == "no evident not" else "no evident not"
    alt_snake = alt.replace(" ", "_")

    label_any = re.compile(r'(?i)\b(?:support|supports?|contrary|contraries)\s*:?\s*')
    tokens = []

    def push_support(s: str):
        s = label_any.sub("", s).strip()
        if s:
            tok = to_snake_phrase(s)
            if tok:
                tokens.append(tok)

    def push_contrary(prefix_snake: str, s: str):
        s = label_any.sub("", s).strip()
        if s:
            tok = prefix_snake + "_" + to_snake_phrase(s)
            tokens.append(tok)

    for line in text.split("\n"):
        line = re.sub(r"^\d+\.\s*", "", line.strip(), flags=re.I)
        line = re.sub(r"^(support|contrary)\s*:?\s*", "", line, flags=re.I)
        if not line:
            continue

        lower_line = line.lower()

        idx = lower_line.find(expected)
        if idx != -1:
            support_part  = line[:idx].strip()
            contrary_part = line[idx + len(expected):].strip()
            if support_part:  push_support(support_part)
            if contrary_part: push_contrary(expected_snake, contrary_part)
            continue

        idx = lower_line.find(alt)
        if idx != -1:
            support_part  = line[:idx].strip()
            contrary_part = line[idx + len(alt):].strip()
            if support_part:  push_support(support_part)
            if contrary_part: push_contrary(alt_snake, contrary_part)
            continue

        push_support(line)

    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " , ".join(out)

def safe_format(ans_raw: str, claim: str) -> str:
    if not USE_POSTFORMAT:
        return ans_raw
    try:
        ans = postformat_aba(ans_raw, claim)
        return ans if ans.strip() else ans_raw
    except Exception as e:
        return f"[POSTFORMAT_ERROR] {type(e).__name__}: {e}\n{ans_raw}"

def sort_df_asc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Topic" not in df.columns and "Claim" in df.columns:
        df = df.rename(columns={"Claim": "Topic"})
    if "Topic" not in df.columns:
        df["Topic"] = ""
    if "ID" not in df.columns:
        return df.sort_values(by=["Topic"], ascending=True, kind="stable")

    df["ID"] = df["ID"].astype(str)
    df["Topic"] = df["Topic"].astype(str)

    df["__id_num"] = pd.to_numeric(df["ID"], errors="coerce")
    df["__id_str"] = df["ID"].astype(str)

    df = df.sort_values(
        by=["__id_num", "__id_str", "Topic"],
        ascending=[True, True, True],
        kind="stable",
        na_position="last",
    )
    return df.drop(columns=["__id_num", "__id_str"], errors="ignore")

KEY_COLS = ["ID", "Topic"]

def run_csv_path_for(out_dir: str, topic_snake: str, run: int) -> str:
    return os.path.join(out_dir, f"{topic_snake}_run{run}.csv")

def load_done_keys(run_csv_path: str, test_col: str) -> set[tuple[str, str]]:
    if not os.path.exists(run_csv_path):
        return set()
    try:
        d = pd.read_csv(run_csv_path)
        d.columns = [str(c).strip() for c in d.columns]
        if "Topic" not in d.columns and "Claim" in d.columns:
            d = d.rename(columns={"Claim": "Topic"})
        if "ID" not in d.columns or "Topic" not in d.columns or test_col not in d.columns:
            return set()

        d["ID"] = d["ID"].astype(str)
        d["Topic"] = d["Topic"].astype(str)
        d[test_col] = d[test_col].fillna("").astype(str)

        d = d.drop_duplicates(subset=KEY_COLS, keep="last")
        d = d[d[test_col].str.strip() != ""]
        return set(zip(d["ID"], d["Topic"]))
    except Exception:
        return set()

def append_csv_rows_sorted(rows: list[dict], path: str):
    """
    Append rows, then rewrite file sorted ASC by ID, then Topic.
    Also drop dup keys (ID,Topic) keep last.
    """
    if not rows:
        return

    df_new = pd.DataFrame(rows)
    if "Topic" not in df_new.columns and "Claim" in df_new.columns:
        df_new = df_new.rename(columns={"Claim": "Topic"})

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.columns = [str(c).strip() for c in df_all.columns]
    if "ID" in df_all.columns:
        df_all["ID"] = df_all["ID"].astype(str)
    if "Topic" in df_all.columns:
        df_all["Topic"] = df_all["Topic"].astype(str)

    if all(c in df_all.columns for c in KEY_COLS):
        df_all = df_all.drop_duplicates(subset=KEY_COLS, keep="last")

    df_all = sort_df_asc(df_all)
    df_all.to_csv(path, index=False)

def rebuild_master_excel(out_dir: str, topic_snake: str, run_tag: str, n_runs: int):
    frames = []
    for run in range(1, n_runs + 1):
        p = run_csv_path_for(out_dir, topic_snake, run)
        if not os.path.exists(p):
            continue
        df_run = pd.read_csv(p)
        df_run.columns = [str(c).strip() for c in df_run.columns]
        if "Topic" not in df_run.columns and "Claim" in df_run.columns:
            df_run = df_run.rename(columns={"Claim": "Topic"})
        test_col = f"Test {run}"
        if test_col not in df_run.columns:
            continue
        keep = [c for c in ["ID","Prompt","Topic",test_col] if c in df_run.columns]
        df_run = df_run[keep].copy()
        df_run["ID"] = df_run["ID"].astype(str)
        df_run["Topic"] = df_run["Topic"].astype(str)
        frames.append(df_run)

    if not frames:
        return None

    master = frames[0].copy()
    for df_run in frames[1:]:
        master = master.merge(df_run, on=["ID","Topic"], how="outer", suffixes=("", "_dup"))
        if "Prompt_dup" in master.columns:
            master["Prompt"] = master["Prompt"].where(
                master["Prompt"].astype(str).str.strip().ne(""),
                master["Prompt_dup"].astype(str)
            )
            master = master.drop(columns=["Prompt_dup"])

    for run in range(1, n_runs + 1):
        col = f"Test {run}"
        if col not in master.columns:
            master[col] = pd.NA
    master["Label"] = ""

    cols = ["ID","Prompt","Topic"] + [f"Test {i}" for i in range(1, n_runs + 1)] + ["Label"]
    for c in cols:
        if c not in master.columns:
            master[c] = ""

    master = master[cols].copy()
    master = sort_df_asc(master)

    master_xlsx = os.path.join(out_dir, f"{topic_snake}_{run_tag}_sheet.xlsx")
    master.to_excel(master_xlsx, index=False)
    return master_xlsx

df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]
required = {TEXT_COL, TOPIC_COL, POSNEG_COL, ID_COL}
miss = [c for c in required if c not in df.columns]
if miss:
    raise SystemExit(f"Missing columns in Excel: {miss}. Found: {list(df.columns)}")

sel_rows = []
sample_set = set(map(str, SAMPLE_IDS)) if SAMPLE_IDS else None
target_norm = norm(TARGET_TOPIC)

for idx, r in df.iterrows():
    text   = r.get(TEXT_COL, "")
    topic  = r.get(TOPIC_COL, "")
    posneg = r.get(POSNEG_COL, "")
    rid    = r.get(ID_COL, None)

    if pd.isna(text):
        continue
    text = (text if isinstance(text, str) else str(text)).strip()
    if not text:
        continue

    if pd.isna(topic) or norm(topic) != target_norm:
        continue

    pol = polarity(posneg or "")
    if pol is None:
        continue

    rid = coerce_id(rid)
    if rid is None or rid == "":
        rid = str(idx + 1)

    if sample_set is not None and str(rid) not in sample_set:
        continue

    claim = f"{pol}_{snake(TARGET_TOPIC)}"
    sel_rows.append({"ID": str(rid), "Text": text, "Claim": claim})

df_sel = pd.DataFrame(sel_rows).reset_index(drop=True)
if (sample_set is None) and isinstance(LIMIT_ROWS, int):
    df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

if df_sel.empty:
    raise SystemExit("No matching rows after filtering/sampling.")

df_sel = df_sel.drop_duplicates(subset=["ID","Claim","Text"], keep="first").reset_index(drop=True)

prepared = []
for _, row in df_sel.iterrows():
    msgs = build_messages(row["Text"], row["Claim"])
    prm  = stringify_messages(msgs)
    prepared.append({
        "ID": row["ID"],
        "Claim": row["Claim"],
        "Messages": msgs,
        "PromptStr": prm
    })

topic_snake = snake(TARGET_TOPIC)
run_tag = f"{snake(MODEL_NAME)}_{'1shot' if USE_ONE_SHOT else '0shot'}"
out_dir = f"ABA_task2_{topic_snake}_{run_tag}"
os.makedirs(out_dir, exist_ok=True)

exp = set((str(it["ID"]), str(it["Claim"])) for it in prepared)
lookup = {(str(it["ID"]), str(it["Claim"])): it for it in prepared}

def key_sort(k):
    rid, claim = k
    return (int(rid) if str(rid).isdigit() else 10**18, claim)

for run in range(1, N_RUNS + 1):
    test_col = f"Test {run}"
    run_csv = run_csv_path_for(out_dir, topic_snake, run)

    done = load_done_keys(run_csv, test_col=test_col)
    missing = exp - done

    print(f"\n========== RUN {run}/{N_RUNS} ==========")
    print(f"[RESUME] file={run_csv}")
    print(f"[RESUME] done={len(done)} missing={len(missing)}")

    if not missing:
        if os.path.exists(run_csv):
            try:
                df_exist = pd.read_csv(run_csv)
                df_exist = sort_df_asc(df_exist)
                df_exist.to_csv(run_csv, index=False)
            except Exception:
                pass
        continue

    out_rows = []
    for (rid, claim) in sorted(missing, key=key_sort):
        it = lookup[(rid, claim)]
        try:
            ans_raw = ask_llm_messages(it["Messages"])
            ans = safe_format(ans_raw, claim)
        except Exception as e:
            ans = f"[ERROR] {type(e).__name__}: {e}"

        out_rows.append({
            "ID": rid,
            "Prompt": it["PromptStr"],
            "Topic": claim,
            test_col: ans
        })

        if PRINT_TO_CONSOLE:
            print(f"[Run {run}] ID={rid} Topic={claim}:\n{ans}\n")

        time.sleep(SLEEP_SEC)

    append_csv_rows_sorted(out_rows, run_csv)

master_xlsx = rebuild_master_excel(out_dir, topic_snake, run_tag, N_RUNS)
print("\n DONE")
print("out_dir     =", out_dir)
print("master_xlsx =", master_xlsx)