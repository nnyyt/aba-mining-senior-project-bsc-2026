import os, time, re, pandas as pd
import google.generativeai as genai

# ================== Gemini config ==================
MODEL_NAME  = "gemini-2.5-pro"
TEMPERATURE = 0
TOP_P       = 0.05


genai.configure(api_key="API KEY HERE")
model = genai.GenerativeModel(MODEL_NAME)

# ================== Minimal config ==================
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
USE_ONE_SHOT = True
SLEEP_SEC = 0.25
DO_RERUN_MISSING = True

ONE_SHOT_EXAMPLES = {
    "good_staff": {
        "text": "The staff were exceptional. So helpful and friendly. Went out of their way for us.",
        "output": "exceptional_staff, helpful_staff, friendly_staff, staff_go_out_of_their_way_for_us, no_evident_not_exceptional_staff, no_evident_not_helpful_staff, no_evident_not_friendly_staff, no_evident_not_staff_go_out_of_their_way_for_us"
    },
    "bad_staff": {
        "text": "The receptionist didn't speak English properly",
        "output": "staff_didn_t_speak_english_properly, have_evident_staff_didn_t_speak_english_properly"
    },
    "good_room": {
        "text": "The room was spacious and comfortable. Very clean with a nice view.",
        "output": "clean_room, well_decorated_room, modern_room, no_evident_not_clean_room, no_evident_not_well_decorated_room, no_evident_not_modern_room"
    },
    "bad_room": {
        "text": "The floor is not cleaned enough, full of dust, water does not drain and floods the room",
        "output": "not_clean_enough_floor, dust_on_floor, water_does_not_drain, floods_room, have_evident_not_clean_enough_floor, have_evident_dust_on_floor, have_evident_water_does_not_drain, have_evident_floods_room"
    },
    "good_facility": {
        "text": "The Hotel itself was very nice, spacious, clean and modern.",
        "output": "nice_hotel, spacious_hotel, clean_hotel, modern_hotel, no_evident_not_nice_hotel, no_evident_not_spacious_hotel, no_evident_not_clean_hotel, no_evident_not_modern_hotel"
    },
    "bad_facility": {
        "text": "when I came but the hotel doesn't have it's own parking lot so sometimes it might be difficult to find a parking spot.",
        "output": "no_own_parking_lot, difficult_to_find_parking, have_evident_no_own_parking_lot, have_evident_difficult_to_find_parking"
    },
    "good_location": {
        "text": "close to the airport, to very clean beach.",
        "output": "close_to_airport, close_to_clean_beach, no_evident_not_close_to_airport, no_evident_not_close_to_clean_beach"
    },
    "bad_location": {
        "text": "we were surprised at the location in a small side street,",
        "output": "located_in_small_side_street, have_evident_located_in_small_side_street"
    },
    "good_food": {
        "text": "Tasty food on the first floor, comfortable restaurant for both cozy evenings and calm work to escape the heat in the midday",
        "output": "tasty_food, comfortable_restaurant_for_cozy_evenings, comfortable_restaurant_for_calm_work, no_evident_not_tasty_food, no_evident_not_comfortable_restaurant_for_cozy_evenings, no_evident_not_comfortable_restaurant_for_calm_work"
    },
    "bad_food": {
        "text": "Ordered pizza one night and was not great, too stodgy",
        "output": "not_great_pizza, stodgy_pizza, have_evident_not_great_pizza, have_evident_stodgy_pizza"
    },
    "good_price": {
        "text": "Great value for money",
        "output": "great_value_for_money, no_evident_not_great_value_for_money"
    },
    "bad_price": {
        "text": "For what you get it was a bit expensive.",
        "output": "expensive_for_what_you_get, have_evident_expensive_for_what_you_get"
    },
    "good_check_in": {
        "text": "clear instructions left access to our room made easy",
        "output": "clear_instructions_to_access_room, no_evident_not_clear_instructions_to_access_room"
    },
    "bad_check_in": {
        "text": "We had a late check in and there was no one in the hotel to give us keys and the door was locked.",
        "output": "no_one_gave_keys_on_late_check_in, door_locked_on_late_check_in, have_evident_no_one_gave_keys_on_late_check_in, have_evident_door_locked_on_late_check_in"
    },
    "good_check_out": {
        "text": "super fast checkin/checkout",
        "output": "fast_check_out, no_evident_not_fast_check_out"
    },
    "bad_check_out": {
        "text": "4. the hotel did not allow to turn over a couple of hours before the delayed plane;",
        "output": "staff_did_not_allow_late_check_out_for_delayed_plane, have_evident_staff_did_not_allow_late_check_out_for_delayed_plane"
    },
    "good_taxi_issue": {
        "text": "We arrived at 2:30 am at the airport, a shuttle taxi to the hotel was waiting for us.",
        "output": "taxi_waiting_for_us, no_evident_not_taxi_waiting_for_us"
    },
    "bad_taxi_issue": {
        "text": "Taxi drivers who cooperate with hotel, both times, from airport and back, charged 2-3 EUR above the price agreed with the hotel (20 EUR). To clarify, it is not about the money, I would certainly give that money as a tip, but feeling that someone is cheating you and stilling money from you is really awkward and ugly, especially when you are alone with small child in the middle of the night and have no alternative way of transport.",
        "output": "charge_above_price_taxi, no_alternative_transport_at_midnight, have_evident_charge_above_price_taxi, have_evident_no_alternative_transport_at_midnight"
    },
    "bad_booking_issue": {
        "text": "However when I booked this property in advance it was a higher price but booking.com reduced the price before my travel date and the hotel refused to accept the lower price which I felt was unfair for me as they charged twice for cancellation and rebooking.",
        "output": "mismatch_cancellation_charge_policy, price_reduced_after_booking, hotel_refused_to_accept_lower_price, have_evident_mismatch_cancellation_charge_policy, have_evident_price_reduced_after_booking, have_evident_hotel_refused_to_accept_lower_price"
    },
}

def norm(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def topic_matches(row_topic: str, target: str) -> bool:
    return norm(row_topic) == norm(target)

def polarity(v: str):
    v = norm(v)
    if v in {"positive", "pos", "good", "p"}: return "good"
    if v in {"negative", "neg", "bad", "n"}:  return "bad"
    return None

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")

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

def get_system_prompt(claim: str) -> str:
    contr = contrary_prefix_for_claim(claim)
    return f"""Generate text in Assumption Based Argumentation (ABA) format from the given text. Use the following conditions carefully.
1. Claim is "{claim}".
2. Supports are written in short words with no adjectives and no adverbs if it is not necessary for understanding.
3. For each support, use vocab from the original text. Do not provide synonyms. Check grammar. Do not provide further opinion.
4. Add a contrary for each support as a new support.
5. Each contrary must use the same word as presented in the support and each contrary starts with "{contr}".
6. Regarding the format of answer, provide a list of all supports and contraries. Do not separate supports and contraries into separated sections. Do not provide assumptions. Do not provide claims."""

def get_example_for_claim(claim: str):
    ex = ONE_SHOT_EXAMPLES.get(norm(claim))
    if not ex:
        return None, None
    return ex["text"], ex["output"]

def build_transcript_prompt(text: str, claim: str, use_one_shot: bool) -> str:
    system = get_system_prompt(claim)
    parts = [f"SYSTEM:\n{system}\n\n---\n"]
    if use_one_shot:
        ex_text, ex_out = get_example_for_claim(claim)
        if ex_text and ex_out:
            parts.append(f"USER:\n[Text] {ex_text}\n\n---\n")
            parts.append(f"ASSISTANT:\n{ex_out}\n\n---\n")
    parts.append(f"USER:\n[Text] {(text or '').strip()}")
    return "".join(parts)

def ask_llm(text: str, claim: str, use_one_shot: bool) -> str:
    prompt = build_transcript_prompt(text, claim, use_one_shot=use_one_shot)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": TEMPERATURE, "top_p": TOP_P},
    )
    return (resp.text or "")

def postformat_for_test_cell(raw: str) -> str:
    # keep it simple: one line
    s = (raw or "").strip()
    s = re.sub(r"\s*\n\s*", " , ", s).strip()
    return s

def ensure_dirs(out_dir: str):
    csv_dir = os.path.join(out_dir, "csv")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return csv_dir, log_dir

def append_csv_rows(rows: list, path: str):
    if not rows:
        return
    df_part = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df_part.to_csv(path, index=False, mode="a", header=write_header)
    
KEY_COLS = ["ID", "Topic"]

def load_done_keys(run_csv_path: str, test_col: str) -> set:
    """
    Return set of (ID,Topic) where test_col is non-empty in run_csv_path.
    """
    if not os.path.exists(run_csv_path):
        return set()
    try:
        d = pd.read_csv(run_csv_path)
        d.columns = [str(c).strip() for c in d.columns]
        if "Topic" not in d.columns and "Claim" in d.columns:
            d = d.rename(columns={"Claim": "Topic"})
        if "ID" not in d.columns or "Topic" not in d.columns:
            return set()
        if test_col not in d.columns:
            return set()

        d["ID"] = d["ID"].astype(str)
        d["Topic"] = d["Topic"].astype(str)
        d[test_col] = d[test_col].fillna("").astype(str)

        d = d.drop_duplicates(subset=KEY_COLS, keep="last")
        d = d[d[test_col].str.strip() != ""]
        return set(zip(d["ID"], d["Topic"]))
    except Exception:
        return set()

def expected_keys(df_sel: pd.DataFrame) -> set:
    d = df_sel[["ID", "Topic"]].copy()
    d["ID"] = d["ID"].astype(str)
    d["Topic"] = d["Topic"].astype(str)
    d = d.drop_duplicates(subset=KEY_COLS, keep="last")
    return set(zip(d["ID"], d["Topic"]))

def first_incomplete_run(df_sel: pd.DataFrame, csv_dir: str, topic_snake: str) -> int:
    exp = expected_keys(df_sel)
    for run in range(1, N_RUNS + 1):
        test_col = f"Test {run}"
        run_csv_path = os.path.join(csv_dir, f"{topic_snake}_run{run}.csv")
        done = load_done_keys(run_csv_path, test_col=test_col)
        if len(exp - done) > 0:
            return run
    return N_RUNS + 1  

def upsert_master_excel(master_xlsx: str, batch_rows: list, cols: list):
    if not batch_rows:
        return

    upd = pd.DataFrame(batch_rows).copy()
    upd.columns = [str(c).strip() for c in upd.columns]
    if "Topic" not in upd.columns and "Claim" in upd.columns:
        upd = upd.rename(columns={"Claim": "Topic"})

    for k in KEY_COLS:
        if k not in upd.columns:
            raise ValueError(f"batch_rows missing required column: {k}")

    upd["ID"] = upd["ID"].astype(str)
    upd["Topic"] = upd["Topic"].astype(str)
    upd = upd.drop_duplicates(subset=KEY_COLS, keep="last")

    if os.path.exists(master_xlsx):
        master = pd.read_excel(master_xlsx)
    else:
        master = pd.DataFrame(columns=cols)

    master.columns = [str(c).strip() for c in master.columns]
    for c in cols:
        if c not in master.columns:
            master[c] = ""

    master["ID"] = master["ID"].astype(str)
    master["Topic"] = master["Topic"].astype(str)
    master = master.drop_duplicates(subset=KEY_COLS, keep="last")

    for c in cols:
        if c in KEY_COLS:
            continue
        master[c] = master[c].astype("string").fillna("")
        if c in upd.columns:
            upd[c] = upd[c].astype("string").fillna("")

    m = master.set_index(KEY_COLS)
    u = upd.set_index(KEY_COLS)

    update_cols = [c for c in u.columns if c in cols and c not in KEY_COLS]

    common = m.index.intersection(u.index)
    if len(common) > 0:
        for c in update_cols:
            newv = u.loc[common, c].astype("string").fillna("")
            mask = newv.str.strip() != ""
            m.loc[common, c] = m.loc[common, c].where(~mask, newv)

    new_idx = u.index.difference(m.index)
    if len(new_idx) > 0:
        add = pd.DataFrame("", index=new_idx, columns=m.columns).astype("string")
        for c in update_cols:
            add[c] = u.loc[new_idx, c].astype("string").fillna("")
        m = pd.concat([m, add], axis=0)

    out = m.reset_index()[cols]
    out.to_excel(master_xlsx, index=False)

df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]

required = {TEXT_COL, TOPIC_COL, POSNEG_COL, ID_COL}
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in Excel: {missing}. Found: {list(df.columns)}")

def build_df_sel(topic_name: str) -> pd.DataFrame:
    rows = []
    sample_set = set(map(str, SAMPLE_IDS)) if SAMPLE_IDS else None

    for idx, r in df.iterrows():
        text   = r.get(TEXT_COL, "")
        topic  = r.get(TOPIC_COL, "")
        posneg = r.get(POSNEG_COL, "")
        rid    = coerce_id(r.get(ID_COL, None))

        if not isinstance(text, str) or not text.strip():
            continue
        if not topic_matches(topic, topic_name):
            continue

        pol = polarity(posneg)
        if pol is None:
            continue

        if rid is None or rid == "":
            rid = str(idx + 1)

        if sample_set is not None and str(rid) not in sample_set:
            continue

        claim = f"{pol}_{snake(topic_name)}"
        rows.append({"ID": str(rid), "Text": text.strip(), "Topic": claim})

    df_sel = pd.DataFrame(rows)
    if df_sel.empty:
        return df_sel

    if (sample_set is None) and isinstance(LIMIT_ROWS, int):
        df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

    df_sel = df_sel.drop_duplicates(subset=["ID", "Topic", "Text"], keep="first").reset_index(drop=True)
    return df_sel

def run_one_topic(topic_name: str):
    df_sel = build_df_sel(topic_name)
    if df_sel.empty:
        print(f"\n[SKIP] No matching rows for topic: {topic_name}")
        return

    topic_snake = snake(topic_name)
    shot_tag = "1shot" if USE_ONE_SHOT else "0shot"
    out_dir = f"/Users/nonny/Downloads/Senior Project 2025/Task2/Gemini/Task2_gemini/ABA_task2_{topic_snake}_gemini_{shot_tag}"
    csv_dir, log_dir = ensure_dirs(out_dir)

    cols = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, N_RUNS + 1)] + ["Label"]
    master_xlsx = os.path.join(out_dir, f"{topic_snake}_gemini_sheet_{shot_tag}.xlsx")

    start_run = first_incomplete_run(df_sel, csv_dir, topic_snake)
    if start_run == N_RUNS + 1:
        print(f"\n[{topic_name}] All runs complete ✅ (nothing missing)")
        return

    for run in range(start_run, N_RUNS + 1):
        test_col = f"Test {run}"
        run_csv_path = os.path.join(csv_dir, f"{topic_snake}_run{run}.csv")
        run_raw_path = os.path.join(log_dir, f"{topic_snake}_run{run}_raw.csv")

        exp = expected_keys(df_sel)
        done = load_done_keys(run_csv_path, test_col=test_col)
        miss = exp - done

        print(f"\n========== TOPIC {topic_name} | RUN {run} / {N_RUNS} ==========")
        print(f"[RESUME] done={len(done)}  missing={len(miss)}  file={run_csv_path}")

        if (not DO_RERUN_MISSING) and len(miss) > 0:
            print("[Info] DO_RERUN_MISSING=False -> skipping rerun for missing")
            continue
        
        df_lookup = df_sel.copy()
        df_lookup["ID"] = df_lookup["ID"].astype(str)
        df_lookup["Topic"] = df_lookup["Topic"].astype(str)

        batch_post, batch_raw = [], []
        new_count = 0

        for (rid, claim) in sorted(miss, key=lambda x: (int(x[0]) if str(x[0]).isdigit() else 10**18, x[1])):
            rr = df_lookup[(df_lookup["ID"] == rid) & (df_lookup["Topic"] == claim)].iloc[0]
            text = rr["Text"]

            prm = build_transcript_prompt(text, claim, use_one_shot=USE_ONE_SHOT)

            err = ""
            raw = ""
            test_val = ""
            try:
                raw = ask_llm(text, claim, use_one_shot=USE_ONE_SHOT) 
                test_val = postformat_for_test_cell(raw)             
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                raw = f"[ERROR] {err}"
                test_val = f"[ERROR] {err}"

            batch_post.append({
                "ID": rid,
                "Prompt": prm,
                "Topic": claim,
                test_col: test_val
            })

            batch_raw.append({
                "ID": rid,
                "Prompt": prm,
                "Topic": claim,
                "Output": raw, 
                "Error": err
            })

            if PRINT_TO_CONSOLE:
                print(f"[Run {run}] ID={rid} Topic={claim}: {test_val}")

            new_count += 1

            if new_count % CHECKPOINT_EVERY == 0:
                append_csv_rows(batch_post, run_csv_path)
                append_csv_rows(batch_raw, run_raw_path)

                excel_batch = [{
                    "ID": br["ID"],
                    "Prompt": br["Prompt"],
                    "Topic": br["Topic"],
                    test_col: br[test_col]
                } for br in batch_post]

                upsert_master_excel(master_xlsx, excel_batch, cols)
                batch_post.clear()
                batch_raw.clear()

            time.sleep(SLEEP_SEC)

        append_csv_rows(batch_post, run_csv_path)
        append_csv_rows(batch_raw, run_raw_path)

        excel_batch = [{
            "ID": br["ID"],
            "Prompt": br["Prompt"],
            "Topic": br["Topic"],
            test_col: br[test_col]
        } for br in batch_post]
        upsert_master_excel(master_xlsx, excel_batch, cols)

    print(f"\n[{topic_name}] DONE")
    print(f"  OUT CSVs : {csv_dir}")
    print(f"  OUT LOGs : {log_dir}")
    print(f"  OUT XLSX : {master_xlsx}")

for tp in TOPICS:
    run_one_topic(tp)