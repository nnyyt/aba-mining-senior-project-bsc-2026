# -*- coding: utf-8 -*-
import os, time, re
import pandas as pd
import requests

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_TAG    = "qwen2.5:7b"
SAFE_TAG     = MODEL_TAG.replace(":", "-")
TEMPERATURE  = 0.0
TOP_P        = 0.05


SRC_XLSX   = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
SHEET      = "Sheet1"
TEXT_COL   = "Selected Content"
TOPIC_COL  = "Topic"
POSNEG_COL = "Pos/Neg"
ID_COL     = "Column1"
BASE_TASK2_DIR = "/Users/nonny/Downloads/Senior Project 2025/Task2/Qwen2.5/Task2_1shot"

TARGET_TOPIC = ""


TOPICS = [
    "room", "facility", "location", "staff", "food", "price",
    "check-in", "check-out", "taxi-issue", "booking-issue"
]

RUN_ALL_TOPICS = True

N_RUNS           = 3
SAMPLE_IDS       = []    
LIMIT_ROWS       = None
PRINT_TO_CONSOLE = True

USE_POSTFORMAT = False    
RESUME         = True
BATCH_SIZE     = 20   
UPDATE_EXCEL_EVERY = 20   

USE_ONE_SHOT = True

ONE_SHOT_EXAMPLES = {
    "good_staff": {
        "text": "The staff were exceptional. So helpful and friendly. Went out of their way for us.",
        "output": "exceptional_staff, helpful_staff, friendly_staff,staff_go_out_of_their_way_for_us, no_evident_not_exceptional_staff, no_evident_not_helpful_staff, no_evident_not_friendly_staff, no_evident_not_staff_go_out_of_their_way_for_us"
    },
    "bad_staff": {
        "text": "The receptionist didn't speak English properly",
        "output": "staff_not_speak_English_properly, have_evident_staff_not_speak_English_properly"
    },
    "good_room": {
        "text": "The room was spacious and comfortable. Very clean with a nice view.",
        "output": "clean_room, well_decorated_room, modern_room, no_evident_not_clean_room, no_evident_not_well_decorated_room, no_evident_not_modern_room"
    },
    "bad_room": {
        "text": "The floor is not cleaned enough, full of dust, water does not drain and floods the room",
        "output": "not_clean_enough_floor, dust_on_floor, water_not_drain, flood_room, have_evident_not_clean_enough_floor, have_evident_dust_on_floor, have_evident_water_not_drain, have_evident_flood_room"
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
        "output": "tasty_food, comfortable_restaurant_for_cozy_evening, comfortable_restaurant_for_calm_work, no_evident_not_tasty_food, no_evident_not_comfortable_restaurant_for_cozy_evening, no_evident_not_comfortable_restaurant_for_calm_work"
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
        "output": "clear_instruction_to_access_room, no_evident_not_clear_instruction_to_access_room"
    },
    "bad_check_in": {
        "text": "We had a late check in and there was no one in the hotel to give us keys and the door was locked.",
        "output": "no_one_give_key_on_late_check-in, door_lock_on_late_check-in, have_evident_no_one_give_key_on_late_check-in, have_evident_door_lock_on_late_check-in"
    },
    "good_check_out": {
        "text": "super fast checkin/checkout",
        "output": "fast_check-out, no_evident_not_fast_check-out"
    },
    "bad_check_out": {
        "text": "4. the hotel did not allow to turn over a couple of hours before the delayed plane;",
        "output": "staff_not_allow_late_check-out_for_delay_plane, have_evident_staff_not_allow_late_check-out_for_delay_plane"
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
        "text": "However when I booked this property in advance it was a higher price but booking.com reduced the price before my travel date and the hotel refused to accept the lower price which I felt was unfair for me as they charged twice for cancellation and rebooking. Maybe this was a booking.com issue or the hotel but I was quite unhappy about this since it was not justified by the owner. I spoke to Ali from booking.com and he said there was no cancellation charge but they charged me",
        "output": "mismatch_cancellation_charge_policy, price_reduce_after_book, hotel_refuse_to_accept_lower_price, have_evident_mismatch_cancellation_charge_policy, have_evident_price_reduce_after_book, have_evident_hotel_refuse_to_accept_lower_price"
    },
}

def norm(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def polarity(v: str):
    v = norm(v)
    if v in {"positive","pos","good","p"}: return "good"
    if v in {"negative","neg","bad","n"}:  return "bad"
    return None

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")

def coerce_id(v):
    try:
        if isinstance(v, float) and v.is_integer(): return int(v)
        if isinstance(v, (int, str)): return int(v)
        return int(v)
    except Exception:
        return v

def contrary_prefix_for_claim(claim: str) -> str:
    c = norm(claim)
    if c.startswith("good_"): return "no evident not"
    if c.startswith("bad_"):  return "have evident"
    raise ValueError(f"Unexpected claim format: '{claim}'")

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

def build_transcript_prompt(text: str, claim: str) -> str:
    system = get_system_prompt(claim)
    parts = [f"SYSTEM:\n{system}\n\n---\n"]
    if USE_ONE_SHOT:
        ex_text, ex_tokens = get_example_for_claim(claim)
        if ex_text and ex_tokens:
            parts.append(f"USER:\n[Text] {ex_text}\n\n---\n")
            parts.append(f"ASSISTANT:\n{ex_tokens}\n\n---\n")
    parts.append(f"USER:\n[Text] {(text or '').strip()}")
    return "".join(parts)

def build_messages(text: str, claim: str) -> list:
    messages = [{"role": "system", "content": get_system_prompt(claim)}]
    if USE_ONE_SHOT:
        ex_text, ex_tokens = get_example_for_claim(claim)
        if ex_text and ex_tokens:
            messages.append({"role": "user", "content": "[Text] " + ex_text})
            messages.append({"role": "assistant", "content": ex_tokens})
    messages.append({"role": "user", "content": "[Text] " + (text or "").strip()})
    return messages

def build_prompt(text: str, claim: str) -> str:
    return build_transcript_prompt(text, claim)

def ask_llm(text: str, claim: str) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_TAG,
        "messages": build_messages(text, claim),
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

def enforce_run_cols(df: pd.DataFrame, test_col: str) -> pd.DataFrame:
    if "Topic" not in df.columns and "Claim" in df.columns:
        df = df.rename(columns={"Claim": "Topic"})
    cols = ["ID", "Prompt", "Topic", test_col]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def enforce_log_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Topic" not in df.columns and "Claim" in df.columns:
        df = df.rename(columns={"Claim": "Topic"})
    cols = ["ID", "Topic", "Output"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def _done_pairs_from_run_csv(run_csv_path: str, test_col: str):
    """Done = Test col non-empty. Pair = (ID, topic_key_lower)."""
    if not os.path.exists(run_csv_path):
        return set()
    try:
        df_old = pd.read_csv(run_csv_path, dtype=str)
        df_old = enforce_run_cols(df_old, test_col)

        df_old["ID"] = pd.to_numeric(df_old["ID"], errors="coerce")
        df_old = df_old.dropna(subset=["ID","Topic"])
        df_old["ID"] = df_old["ID"].astype(int)
        df_old["_tk"] = df_old["Topic"].astype(str).str.strip().str.lower()

        ok = df_old[test_col].notna() & (df_old[test_col].astype(str).str.strip() != "")
        df_old = df_old[ok]
        return set(zip(df_old["ID"], df_old["_tk"]))
    except Exception:
        return set()

def _done_pairs_from_log_csv(run_log_path: str):
    """Done = Output non-empty. Pair = (ID, topic_key_lower)."""
    if not os.path.exists(run_log_path):
        return set()
    try:
        df_old = pd.read_csv(run_log_path, dtype=str)
        df_old = enforce_log_cols(df_old)

        df_old["ID"] = pd.to_numeric(df_old["ID"], errors="coerce")
        df_old = df_old.dropna(subset=["ID","Topic"])
        df_old["ID"] = df_old["ID"].astype(int)
        df_old["_tk"] = df_old["Topic"].astype(str).str.strip().str.lower()

        ok = df_old["Output"].notna() & (df_old["Output"].astype(str).str.strip() != "")
        df_old = df_old[ok]
        return set(zip(df_old["ID"], df_old["_tk"]))
    except Exception:
        return set()

def save_master_excel(df_sel, out_dir, run_base, n_runs):
    master_xlsx = os.path.join(out_dir, f"{run_base}_sheet.xlsx")
    csv_dir = os.path.join(out_dir, "csv")
    log_dir = os.path.join(out_dir, "logs")

    old_labels = {}
    if os.path.exists(master_xlsx):
        try:
            prev = pd.read_excel(master_xlsx)
            if {"ID","Topic","Label"}.issubset(prev.columns):
                tmp = prev[["ID","Topic","Label"]].copy()
                tmp["ID"] = pd.to_numeric(tmp["ID"], errors="coerce")
                tmp = tmp.dropna(subset=["ID","Topic"])
                tmp["ID"] = tmp["ID"].astype(int)
                tmp["_tk"] = tmp["Topic"].astype(str).str.strip().str.lower()
                for _, r in tmp.iterrows():
                    old_labels[(int(r["ID"]), str(r["_tk"]))] = r["Label"]
        except Exception:
            pass

    wide = pd.DataFrame({
        "ID": df_sel["ID"],
        "Prompt": [build_prompt(t, c) for t, c in zip(df_sel["Text"], df_sel["Claim"])],
        "Topic": df_sel["Claim"],
    })
    wide["ID"] = pd.to_numeric(wide["ID"], errors="coerce")
    wide = wide.dropna(subset=["ID","Topic"])
    wide["ID"] = wide["ID"].astype(int)
    wide["_tk"] = wide["Topic"].astype(str).str.strip().str.lower()
    wide = wide.drop_duplicates(subset=["ID","_tk"], keep="last")

    for i in range(1, n_runs + 1):
        test_col = f"Test {i}"
        run_csv_path = os.path.join(csv_dir, f"{run_base}_run{i}.csv")
        run_log_path = os.path.join(log_dir, f"{run_base}_run{i}.csv")

        wide[test_col] = ""

        if os.path.exists(run_csv_path):
            try:
                rdf = pd.read_csv(run_csv_path, dtype=str)
                rdf = enforce_run_cols(rdf, test_col)
                rdf["ID"] = pd.to_numeric(rdf["ID"], errors="coerce")
                rdf = rdf.dropna(subset=["ID","Topic"])
                rdf["ID"] = rdf["ID"].astype(int)
                rdf["_tk"] = rdf["Topic"].astype(str).str.strip().str.lower()
                rdf = rdf.drop_duplicates(subset=["ID","_tk"], keep="last")
                rdf = rdf[["ID","_tk", test_col]]
                wide = wide.merge(rdf, on=["ID","_tk"], how="left", suffixes=("", "_r"))
                if f"{test_col}_r" in wide.columns:
                    wide[test_col] = wide[test_col].where(
                        wide[test_col].notna() & (wide[test_col].astype(str).str.strip() != ""),
                        wide[f"{test_col}_r"]
                    )
                    wide = wide.drop(columns=[f"{test_col}_r"])
            except Exception:
                pass

        if os.path.exists(run_log_path):
            try:
                ldf = pd.read_csv(run_log_path, dtype=str)
                ldf = enforce_log_cols(ldf)
                ldf["ID"] = pd.to_numeric(ldf["ID"], errors="coerce")
                ldf = ldf.dropna(subset=["ID","Topic"])
                ldf["ID"] = ldf["ID"].astype(int)
                ldf["_tk"] = ldf["Topic"].astype(str).str.strip().str.lower()
                ldf = ldf.drop_duplicates(subset=["ID","_tk"], keep="last")
                ldf = ldf.rename(columns={"Output": f"{test_col}__log"})
                ldf = ldf[["ID","_tk", f"{test_col}__log"]]
                wide = wide.merge(ldf, on=["ID","_tk"], how="left")
                wide[test_col] = wide[test_col].where(
                    wide[test_col].notna() & (wide[test_col].astype(str).str.strip() != ""),
                    wide[f"{test_col}__log"]
                )
                wide = wide.drop(columns=[f"{test_col}__log"])
            except Exception:
                pass

        wide[test_col] = wide[test_col].fillna("")

    wide["Label"] = [old_labels.get((int(i), str(tk)), "") for i, tk in zip(wide["ID"], wide["_tk"])]

    cols = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, n_runs + 1)] + ["Label"]
    for c in cols:
        if c not in wide.columns:
            wide[c] = ""
    wide = wide[cols].drop_duplicates(subset=["ID","Topic"], keep="last")

    wide.to_excel(master_xlsx, index=False)
    if PRINT_TO_CONSOLE:
        print(f"[Excel] Updated: {master_xlsx}")

def run_for_topic(topic_name: str):
    df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    required = {TEXT_COL, TOPIC_COL, POSNEG_COL, ID_COL}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in Excel: {missing}. Found: {list(df.columns)}")

    sample_set = set(SAMPLE_IDS) if SAMPLE_IDS else None
    sel = []
    for idx, r in df.iterrows():
        text   = r.get(TEXT_COL, "")
        topic  = r.get(TOPIC_COL, "")
        posneg = r.get(POSNEG_COL, "")
        rid    = coerce_id(r.get(ID_COL, None))

        if not isinstance(text, str) or not text.strip():
            continue
        if norm(topic) != norm(topic_name):
            continue

        pol = polarity(posneg)
        if pol is None:
            continue

        if rid in (None, "") or (isinstance(rid, float) and pd.isna(rid)):
            rid = idx + 1

        if sample_set is not None and (rid not in sample_set):
            continue

        claim = f"{pol}_{snake(topic_name)}"
        sel.append({"ID": rid, "Text": text.strip(), "Claim": claim})

    df_sel = pd.DataFrame(sel).reset_index(drop=True)
    if (sample_set is None) and isinstance(LIMIT_ROWS, int):
        df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

    if df_sel.empty:
        print(f"[{topic_name}] No matching rows after filtering.")
        return

    topic_snake = snake(topic_name)
    run_base = f"{topic_snake}_{SAFE_TAG}_1shot"
    out_dir  = os.path.join(BASE_TASK2_DIR, f"ABA_task2_{topic_snake}_{SAFE_TAG}_1shot")
    csv_dir  = os.path.join(out_dir, "csv")
    log_dir  = os.path.join(out_dir, "logs")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for run in range(1, N_RUNS + 1):
        test_col = f"Test {run}"
        run_csv = os.path.join(csv_dir, f"{run_base}_run{run}.csv")  
        run_log = os.path.join(log_dir, f"{run_base}_run{run}.csv") 

        if RESUME and os.path.exists(run_csv):
            try:
                tmp = pd.read_csv(run_csv, dtype=str)
                tmp = enforce_run_cols(tmp, test_col)
                tmp.to_csv(run_csv, index=False)
            except Exception:
                pass

        done_pairs = set()
        if RESUME:
            done_pairs |= _done_pairs_from_run_csv(run_csv, test_col)
            done_pairs |= _done_pairs_from_log_csv(run_log)

        if PRINT_TO_CONSOLE:
            print(f"\n[{topic_name}] ========= RUN {run} / {N_RUNS} =========")
            print(f"[{topic_name}] run_csv = {run_csv}")
            print(f"[{topic_name}] run_log = {run_log}")
            if done_pairs:
                print(f"[{topic_name}] Resuming: {len(done_pairs)} already done (ID+Topic).")

        rows_batch = []
        results_csv_exists = os.path.exists(run_csv)
        log_csv_exists = os.path.exists(run_log)

        since_excel = 0

        for _, rr in df_sel.iterrows():
            rid = int(rr["ID"])
            claim = str(rr["Claim"]).strip()
            pair = (rid, norm(claim))

            if pair in done_pairs:
                if PRINT_TO_CONSOLE:
                    print(f"[{topic_name}] [Run {run}] ID={rid} Claim={claim}: skipped")
                continue

            prm = build_prompt(rr["Text"], claim)
            ans_raw = ask_llm(rr["Text"], claim)
            ans = postformat_aba(ans_raw) if USE_POSTFORMAT else ans_raw

            df_log = pd.DataFrame([{"ID": rid, "Topic": claim, "Output": ans_raw}])
            df_log = enforce_log_cols(df_log)
            df_log.to_csv(
                run_log,
                index=False,
                mode=("a" if log_csv_exists else "w"),
                header=(not log_csv_exists),
            )
            log_csv_exists = True
            
            rows_batch.append({"ID": rid, "Prompt": prm, "Topic": claim, test_col: ans})
            done_pairs.add(pair)

            if PRINT_TO_CONSOLE:
                print(f"[{topic_name}] [Run {run}] ID={rid}: done")

            if len(rows_batch) >= BATCH_SIZE:
                part = pd.DataFrame(rows_batch)
                part = enforce_run_cols(part, test_col)
                part.to_csv(
                    run_csv,
                    index=False,
                    mode=("a" if results_csv_exists else "w"),
                    header=(not results_csv_exists),
                )
                results_csv_exists = True
                rows_batch.clear()

            since_excel += 1
            if UPDATE_EXCEL_EVERY and since_excel >= UPDATE_EXCEL_EVERY:
                save_master_excel(df_sel, out_dir, run_base, N_RUNS)
                since_excel = 0

            time.sleep(0.25)

        if rows_batch:
            part = pd.DataFrame(rows_batch)
            part = enforce_run_cols(part, test_col)
            part.to_csv(
                run_csv,
                index=False,
                mode=("a" if results_csv_exists else "w"),
                header=(not results_csv_exists),
            )
            rows_batch.clear()

        if os.path.exists(run_csv):
            try:
                rdf = pd.read_csv(run_csv, dtype=str)
                rdf = enforce_run_cols(rdf, test_col)
                rdf["ID"] = pd.to_numeric(rdf["ID"], errors="coerce")
                rdf = rdf.dropna(subset=["ID","Topic"])
                rdf["ID"] = rdf["ID"].astype(int)
                rdf["_tk"] = rdf["Topic"].astype(str).str.strip().str.lower()
                rdf = rdf.drop_duplicates(subset=["ID","_tk"], keep="last").drop(columns=["_tk"])
                rdf = rdf.sort_values("ID")
                rdf.to_csv(run_csv, index=False)
            except Exception:
                pass

        save_master_excel(df_sel, out_dir, run_base, N_RUNS)

    print(f"\n[{topic_name}] DONE.")
    print(f"Folder: {out_dir}")
    print(f"Sheet:  {os.path.join(out_dir, f'{run_base}_sheet.xlsx')}")

if __name__ == "__main__":
    if RUN_ALL_TOPICS:
        for t in TOPICS:
            run_for_topic(t)
    else:
        run_for_topic(TARGET_TOPIC)