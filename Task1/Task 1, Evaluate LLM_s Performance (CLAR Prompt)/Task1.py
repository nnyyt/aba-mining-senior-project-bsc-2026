import os
import json
import time
import pandas as pd
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_TAG   = os.getenv("MODEL_TAG", "deepseek-r1:7b")

SRC_XLSX = "/Users/nonny/Downloads/Senior Project 2025/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
SHEET    = "Sheet2"

SAMPLE_IDS = []
SHOT_SET = [0,1,2,3,4,5] 

OUT_ROOT = f"/Users/nonny/Downloads/Senior Project 2025/Task1/Task1_{MODEL_TAG.replace(':','-')}"

TEMPERATURE = 0.0
TOP_P = 0.05
SLEEP_BETWEEN_CALLS = 0.0

MAX_RAW_CHARS = 0        
CHUNK_SIZE = 20           
RUNS_PER_SHOT = 3        

df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)

if "Column1" in df.columns and "ID" not in df.columns:
    df = df.rename(columns={"Column1": "ID"})

def _not_empty_series(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")

def _coerce_id(v):
    try:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return v
    except Exception:
        return v
    
df_pos = pd.DataFrame()
df_neg = pd.DataFrame()

if "PositiveReview" in df.columns:
    df_pos = df[["ID", "PositiveReview"]].rename(columns={"PositiveReview": "Review"})
    df_pos = df_pos[_not_empty_series(df_pos["Review"])].drop_duplicates()

if "NegativeReview" in df.columns:
    df_neg = df[["ID", "NegativeReview"]].rename(columns={"NegativeReview": "Review"})
    df_neg = df_neg[_not_empty_series(df_neg["Review"])].drop_duplicates()

df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
df_combined = df_combined.sort_values(by=["ID"]).reset_index(drop=True)

if SAMPLE_IDS:
    sample_set = set(_coerce_id(x) for x in SAMPLE_IDS)
    df_combined["ID_coerced"] = df_combined["ID"].apply(_coerce_id)
    df_combined = df_combined[df_combined["ID_coerced"].isin(sample_set)]
    df_combined = df_combined.drop(columns=["ID_coerced"]).reset_index(drop=True)

# ===================== Few-shot examples =====================
SYSTEM_PROMPT = (
    'Please output the following [text] according to the [constraints] in the [output format].\n '
    '[constraints]* The output should only be in the [output format], and you must classify which part of the text corresponds to which Topic in the [Topics]. '
    'Additionally, determine whether each classified element is Positive or Negative. If there is no corresponding element, put Null for both `text` and `label`. '
    'The most important constraint is not to include any extra characters such as newline characters, `json`, or backticks, or any other unnecessary text outside of the [output format]. '
    'If there are two or more elements of the same Topic, number each so that they do not conflict when converted to json format data. '
    'However, if they have the same NegPos label, keep them in one Text as much as possible.* \n '
    '[Topics] Room, Staff, Location, Food, Price, Facility, Check-in, Check-out, Taxi-issue, Booking-issue, Off \n\n '
    '[output format] '
    '{"Topics":[{"Room":[{"text": "test","label": "Positive"}],'
    '"Staff":[{"text": null,"label": null}],'
    '"Location":[{"text": "test","label": "Positive"}],'
    '"Food":[{"text": "test","label": "Negative"}],'
    '"Price":[{"text": "test","label": "Positive"}],'
    '"Facility":[{"text": "test","label": "Negative"}],'
    '"Check-in":[{"text": "test","label": "Positive"}],'
    '"Check-out":[{"text": null,"label": null}],'
    '"Taxi-issue":[{"text": null,"label": null}],'
    '"Booking-issue":[{"text": null,"label": null}],'
    '"Off":[{"text": null,"label": null}]}]}'
)

FEW_SHOT_EXAMPLES = [
    # Example 1
    {
        "user": "The room is enough big. But the room was a little bit durty.",
        "assistant": (
            '{"Topics":[{"Room1":[{"text": "The room is enough big.","label": "Positive"}],'
            '"Room2":[{"text": "the room was a little bit durty.","label": "Negative"}],'
            '"Staff":[{"text": null,"label": null}],'
            '"Location":[{"text": null,"label": null}],'
            '"Food":[{"text": null,"label": null}],'
            '"Price":[{"text": null,"label": null}],'
            '"Facility":[{"text": null,"label": null}],'
            '"Check-in":[{"text": null,"label": null}],'
            '"Check-out":[{"text": null,"label": null}],'
            '"Taxi-issue":[{"text": null,"label": null}],'
            '"Booking-issue":[{"text": null,"label": null}],'
            '"Off":[{"text": null,"label": null}]}]}'
        ),
    },
    # Example 2
    {
        "user": "The room was very clean cheap, well decorated and modern, although not big.",
        "assistant": (
            '{"Topics":[{"Room1":[{"text": "The room was very clean, well decorated and modern","label": "Positive"}],'
            '"Room2":[{"text": "although not big","label": "Negative"}],'
            '"Price":[{"text": "cheap","label": "Positive"}],'
            '"Staff":[{"text": null,"label": null}],'
            '"Location":[{"text": null,"label": null}],'
            '"Food":[{"text": null,"label": null}],'
            '"Facility":[{"text": null,"label": null}],'
            '"Check-in":[{"text": null,"label": null}],'
            '"Check-out":[{"text": null,"label": null}],'
            '"Taxi-issue":[{"text": null,"label": null}],'
            '"Booking-issue":[{"text": null,"label": null}],'
            '"Off":[{"text": null,"label": null}]}]}'
        ),
    },
    # Example 3
    {
        "user": ("Location. The hotel was new and close to the airport, which made traveling easy. However, there was a lot of street noise outside the window. "
                 "Staff. The receptionist was polite and friendly. However, check-in took longer than expected. "
                 "The hotel lobby was welcoming and spacious. The room had a comfortable bed, but the air conditioning was loud at night. "
                 "The neighbors were noisy through the walls, and the WiFi in the room was weak and unreliable. "
                 "The breakfast buffet was delicious; however, the coffee was terrible. The price was reasonable for the quality. "
                 "The building was charming with historical architecture."),
        "assistant": (
            '{"Topics":[{"Room1":[{"text": "The room had a comfortable bed.","label": "Positive"}],'
            '"Room2":[{"text": "The air conditioning was loud at night.","label": "Negative"}],'
            '"Room3":[{"text": "The neighbors were noisy through the walls.","label": "Negative"}],'
            '"Room4":[{"text": "the WiFi in the room was weak and unreliable.","label": "Negative"}],'
            '"Staff":[{"text": "The receptionist was polite and friendly.","label": "Positive"}],'
            '"Location1":[{"text": "close to the airport, which made traveling easy.","label": "Positive"}],'
            '"Location2":[{"text": "there was a lot of street noise outside the window.","label": "Negative"}],'
            '"Food1":[{"text": "The breakfast buffet was delicious.","label": "Positive"}],'
            '"Food2":[{"text": "the coffee was terrible.","label": "Negative"}],'
            '"Price":[{"text": "The price was reasonable for the quality.","label": "Positive"}],'
            '"Facility1":[{"text": "The hotel was new.","label": "Positive"}],'
            '"Facility2":[{"text": "The hotel lobby was welcoming and spacious.","label": "Positive"}],'
            '"Facility3":[{"text": "The building was charming with historical architecture.","label": "Positive"}],'
            '"Check-in":[{"text": "check-in took longer than expected.","label": "Negative"}],'
            '"Check-out":[{"text": null,"label": null}],'
            '"Taxi-issue":[{"text": null,"label": null}],'
            '"Booking-issue":[{"text": null,"label": null}],'
            '"Off":[{"text": "Location. Staff.","label": null}]}]}'
        ),
    },
    # Example 4
    {
        "user": "location, service, overall was good, Sure worth it to come back again",
        "assistant": (
            '{"Topics":[{"Room":[{"text": null,"label": null}],'
            '"Staff":[{"text": null,"label": null}],'
            '"Location":[{"text": null,"label": null}],'
            '"Food":[{"text": null,"label": null}],'
            '"Price":[{"text": null,"label": null}],'
            '"Facility":[{"text": null,"label": null}],'
            '"Check-in":[{"text": null,"label": null}],'
            '"Check-out":[{"text": null,"label": null}],'
            '"Taxi-issue":[{"text": null,"label": null}],'
            '"Booking-issue":[{"text": null,"label": null}],'
            '"Off":[{"text": "location, service, overall was good, Sure worth it to come back again","label": "Null"}]}]}'
        ),
    },
    # Example 5
    {
        "user": "The apartment was new. The breakfast was amazing and the price was quite reasonable. Overall, we definitely planning to return again soon!",
        "assistant": (
            '{"Topics":[{"Room":[{"text": null,"label": null}],'
            '"Staff":[{"text": null,"label": null}],'
            '"Location":[{"text": null,"label": null}],'
            '"Food":[{"text": "The breakfast was amazing and the price was quite reasonable.","label": "Positive"}],'
            '"Price":[{"text": null,"label": null}],'
            '"Facility":[{"text": "The apartment was new.","label": "Positive"}],'
            '"Check-in":[{"text": null,"label": null}],'
            '"Check-out":[{"text": null,"label": null}],'
            '"Taxi-issue":[{"text": null,"label": null}],'
            '"Booking-issue":[{"text": null,"label": null}],'
            '"Off":[{"text": "Overall, we definitely planning to return again soon!","label": "Positive"}]}]}'
        ),
    },
]

def build_messages(review_text: str, shot_k: int):
    """
    Build messages with system prompt + first k examples + real input.
    shot_k = 0..5
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    k = max(0, min(shot_k, len(FEW_SHOT_EXAMPLES)))
    for i in range(k):
        ex = FEW_SHOT_EXAMPLES[i]
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": str(review_text or "")})
    return messages

def build_full_prompt(messages):
    return "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

def parse_response(response_json, review_id, full_prompt):
    results = []
    try:
        parsed = json.loads(response_json)
        topics_list = parsed["Topics"]
        for topic_group in topics_list:
            for topic, entries in topic_group.items():
                for entry in entries:
                    text = entry.get("text") or ""
                    label = entry.get("label") or ""
                    results.append({
                        "ID": review_id,
                        "FullPrompt": full_prompt,
                        "Topics": topic,
                        "Text": text,
                        "NegPos": label
                    })
    except Exception:
        pass
    return results

def ollama_chat(messages):
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_TAG,
        "messages": messages,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
    }
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if content is None:
        raise RuntimeError("No 'message.content' found in Ollama response.")
    return content

def _ids_from_log_jsonl(log_path: str):
    """Collect IDs from existing JSONL log (if present)."""
    ids = set()
    if not os.path.exists(log_path):
        return ids
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        ids.add(obj["id"])
                except Exception:
                    continue
    except Exception:
        pass
    return ids

def _ids_from_csv(csv_path: str):
    """Collect IDs already written to the per-run CSV (if present)."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df_old = pd.read_csv(csv_path)
        if "ID" in df_old.columns:
            return set(df_old["ID"].dropna().tolist())
    except Exception:
        return set()
    return set()

def _append_rows_to_csv(rows, csv_path):
    if not rows:
        return
    df_chunk = pd.DataFrame(rows, columns=["ID", "FullPrompt", "Topics", "Text", "NegPos"])
    write_header = not os.path.exists(csv_path)
    df_chunk.to_csv(csv_path, mode="a", header=write_header, index=False)

def _dedupe_csv(csv_path: str):
    """Drop duplicate rows (ID, Topics, Text, NegPos) to keep CSV clean when resuming."""
    if not os.path.exists(csv_path):
        return
    try:
        df_old = pd.read_csv(csv_path)
        if not df_old.empty:
            df_old = df_old.drop_duplicates(subset=["ID", "Topics", "Text", "NegPos"], keep="last")
            df_old.to_csv(csv_path, index=False)
    except Exception:
        pass

def run_for_shot(shot_k: int):
    out_dir = os.path.join(OUT_ROOT, f"{shot_k}-shot")
    log_dir = os.path.join(out_dir, "logs")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print("="*80)
    print(f"START: {shot_k}-shot   Model={MODEL_TAG}   Host={OLLAMA_HOST}")
    print("="*80)

    for run_num in range(1, RUNS_PER_SHOT + 1):
        final_results = []
        buffer_results = []

        out_csv = os.path.join(csv_dir, f"output_run_{run_num}.csv")
        log_path = os.path.join(log_dir, f"log_run_{run_num}.jsonl")

        processed_ids = set()
        processed_ids |= _ids_from_csv(out_csv)
        processed_ids |= _ids_from_log_jsonl(log_path)

        with open(log_path, "a", encoding="utf-8") as logf:
            for _, row in df_combined.iterrows():
                review_id = row["ID"]
                if review_id in processed_ids:
                    continue

                review_text = row["Review"]

                try:
                    messages = build_messages(review_text, shot_k=shot_k)
                    full_prompt = build_full_prompt(messages)
                    response_content = ollama_chat(messages)

                    if MAX_RAW_CHARS and len(response_content) > MAX_RAW_CHARS:
                        print(f"ID={review_id}\n{response_content[:MAX_RAW_CHARS]}\n")
                    else:
                        print(f"ID={review_id}\n{response_content}\n")

                    log_obj = {
                        "run": run_num,
                        "shot": shot_k,
                        "id": int(review_id) if pd.notna(review_id) else review_id,
                        "prompt_messages": messages,
                        "raw_response": response_content
                    }
                    logf.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

                    parsed_rows = parse_response(response_content, review_id, full_prompt)
                    final_results.extend(parsed_rows)
                    buffer_results.extend(parsed_rows)

                    if len(buffer_results) >= CHUNK_SIZE:
                        _append_rows_to_csv(buffer_results, out_csv)
                        print(f"[Shot {shot_k} Run {run_num}] Checkpoint saved ({len(buffer_results)} rows) → {out_csv}")
                        buffer_results = []

                except Exception as e:
                    err_obj = {
                        "run": run_num,
                        "shot": shot_k,
                        "id": int(review_id) if pd.notna(review_id) else review_id,
                        "error": str(e),
                        "prompt_messages": build_messages(review_text, shot_k=shot_k),
                    }
                    logf.write(json.dumps(err_obj, ensure_ascii=False) + "\n")

                if SLEEP_BETWEEN_CALLS > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS)

        if buffer_results:
            _append_rows_to_csv(buffer_results, out_csv)
            print(f"[Shot {shot_k} Run {run_num}] Final chunk saved ({len(buffer_results)} rows) → {out_csv}")

        _dedupe_csv(out_csv)

        print(f"[Shot {shot_k} Run {run_num}] Log: {log_path}")
        print(f"[Shot {shot_k} Run {run_num}] CSV: {out_csv}")

    print(f"\n🎯 Finished {shot_k}-shot. Outputs under:\n  - Logs: {os.path.join(OUT_ROOT, f'{shot_k}-shot', 'logs')}\n  - CSVs: {os.path.join(OUT_ROOT, f'{shot_k}-shot', 'csv')}\n")

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    for shot_k in SHOT_SET:
        run_for_shot(shot_k)
    print("All requested shots completed.")
