import os, time, re, json, pandas as pd
import requests

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_TAG    = "deepseek-r1:7b"
SAFE_TAG     = MODEL_TAG.replace(":", "-")
TEMPERATURE  = 0.0
TOP_P        = 0.05
# TIMEOUT_SEC = 180 

SRC_XLSX = "C:/Users/9888tawan/Downloads/Task 2, Evaluate LLM's Performance (CLAR Prompt)-20251021T131753Z-1-001/Task 2, Evaluate LLM_s Performance (CLAR Prompt)/Original ABA Dataset for Version 2 (25-07-2025 Nonny Version).xlsx"
SHEET      = "Nonny Version"
TEXT_COL   = "Selected Content"
TOPIC_COL  = "Topic"
POSNEG_COL = "Pos/Neg"
ID_COL     = "Column1"
TARGET_TOPIC = ""
N_RUNS       = 3
SAMPLE_IDS   = []     
LIMIT_ROWS   = None
PRINT_TO_CONSOLE = True
USE_POSTFORMAT   = True
USE_ONE_SHOT     = True
CHUNK_SIZE = 20

TOPICS = [
    "room", "facility", "location", "staff", "food", "price",
    "check-in", "check-out", "taxi-issue", "booking-issue"
]

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
    return True if target is None else norm(row_topic) == norm(target)
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
        if isinstance(v, (int, str)): return v
        return int(v)
    except Exception: return v
def contrary_prefix_for_claim(claim: str) -> str:
    c = norm(claim)
    if c.startswith("good_"): return "no evident not"
    if c.startswith("bad_"):  return "have evident"
    raise ValueError(f"Unexpected claim format: '{claim}' - must start with 'good_' or 'bad_'")
def get_example_for_claim(claim: str):
    return ONE_SHOT_EXAMPLES.get(norm(claim), {}).get("text"), ONE_SHOT_EXAMPLES.get(norm(claim), {}).get("output")

def system_prompt_for_claim(claim: str) -> str:
    contr = contrary_prefix_for_claim(claim)
    return (
        "Generate text in Assumption Based Argumentation (ABA) format from the given text. Use the following conditions carefully.\n\n"
        f"1. Claim is \"{claim}\".\n"
        "2. Supports are written in short words with no adjectives and no adverbs if it is not necessary for understanding.\n"
        "3. For each support, use vocab from the original text. Do not provide synonyms. Check grammar. Do not provide further opinion.\n"
        "4. Add a contrary for each support as a new support.\n"
        f"5. Each contrary must use the same word as presented in the support and each contrary starts with \"{contr}\".\n"
        "6. Regarding the format of answer, provide a list of all supports and contraries. Do not separate supports and contraries into separated sections. Do not provide assumptions. Do not provide claims.\n\n"
        "[Text]"
    )
def build_messages(text: str, claim: str):
    msgs = [{"role": "system", "content": system_prompt_for_claim(claim)}]
    if USE_ONE_SHOT:
        ex_text, ex_output = get_example_for_claim(claim)
        if ex_text and ex_output:
            msgs.append({"role": "user", "content": ex_text})
            msgs.append({"role": "assistant", "content": ex_output})
    msgs.append({"role": "user", "content": (text or "").strip()})
    return msgs
def stringify_messages(msgs: list) -> str:
    role_map = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
    return "\n\n---\n\n".join(f"{role_map.get(m['role'], m['role']).upper()}:\n{(m.get('content') or '').strip()}" for m in msgs)
def ask_llm_messages(messages: list) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_TAG,
        "messages": messages,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
    }
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message", {}).get("content") or "").strip()
        if not content:
            return "[ERROR] Empty response from model (timeout/blank). Skipping."
        return content
    except requests.exceptions.Timeout:
        return "[ERROR] Timeout after 180s while waiting for model response. Skipping."
    except requests.exceptions.RequestException as e:
        return f"[ERROR] RequestException: {e}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def postformat_aba(raw: str, claim: str) -> str:
    """
    Robust ABA post-formatter.
    - Handles: Support/Contrary labels (bold/plain, with/without []), semicolon pairs,
      bold sentences, header blocks ("Supports:" / "Contraries:" with CSV or bullet lists),
      two-line unlabeled pairs, lone snake tokens, numbered support/contrary pairs,
      arrow notation (1. support -> contrary), parentheticals, and mixed spacing like "no evident not_X".
    - More permissive: accepts supports up to 12 words / 120 chars.
    - Fallback: if nothing parsed, extract any snake_case tokens and well-formed contraries.
    """
    if not isinstance(raw, str):
        raw = str(raw or "")
    txt = raw.strip()
    if not txt:
        return ""
    
    txt = re.sub(r'(?is)<think>.*?</think>', '', txt)
    txt = re.sub(r'```\w*', '', txt).replace('```', '')
    
    lines_all = txt.split('\n')
    lines_for_this_id = []
    for ln in lines_all:
        if re.match(r'^ID=\d+', ln.strip()):
            if lines_for_this_id: 
                break
            else: 
                continue
        lines_for_this_id.append(ln)
    
    txt = '\n'.join(lines_for_this_id)
    
    c = (claim or "").strip().lower()
    expected_prefix = "have_evident" if c.startswith("bad_") else "no_evident_not"
    
    TOKEN = re.compile(r'^[a-z0-9]{2,}(?:_[a-z0-9]+)*$') 

    CONTRA = re.compile(r'^(?:no_evident_not|have_evident)_[a-z0-9]+(?:_[a-z0-9]+)*$')
    
    EXCLUDED_TOKENS = {
        'support', 'supports', 'contrary', 'contraries',
        'support_1', 'support_2', 'support_3', 'support_4', 'support_5',
        'contrary_1', 'contrary_2', 'contrary_3', 'contrary_4', 'contrary_5',
        'here_is_the_text_in_assumption', 'based_on', 'the_text',
        'assumption_based_argumentation', 'aba_format', 'claim'
    }
    
    def _clean_content(text: str) -> str:
        if not text:
            return ""
        s = str(text)
        s = s.replace("```", "")
        s = re.sub(r'^["\'`*\s]+|["\'`*\s.,;:]+$', '', s.strip())
        s = re.sub(r'(?i)\bhave\s+evident\b', 'have_evident', s)
        s = re.sub(r'(?i)\bno\s+evident\s+not\b', 'no_evident_not', s)
        s = re.sub(r'\s+', '_', s.lower()).strip('_')
        return s
    
    def strip_parentheticals(s: str) -> str:
        """Remove parenthetical notes like (highly recommended) or (from previous context)"""
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
        return s.strip()
    
    def support_payload_to_token(s: str) -> str:
        s_raw = s.strip().strip('"\'* `')
        s_raw = strip_parentheticals(s_raw)
        if len(s_raw) > 120:
            return ""
        if len(re.findall(r'\w+', s_raw)) > 12:
            return ""
        tok = _clean_content(s_raw)
        return tok if TOKEN.match(tok) and tok.count('_') <= 12 and tok not in EXCLUDED_TOKENS else ""
    
    def normalize_contrary_payload(s: str) -> str:
        s0 = s.strip().strip('"\'* `')
        s0 = strip_parentheticals(s0)
        s0 = re.sub(r'(?i)\bhave\s+evident\b(?=[_\s:;.,-]|$)', 'have_evident', s0)
        s0 = re.sub(r'(?i)\bno\s+evident\s+not\b(?=[_\s:;.,-]|$)', 'no_evident_not', s0)
        s0 = re.sub(r'(?i)\bno\s+evident\s+not(?=\s*_)\b', 'no_evident_not', s0)
        s_snake = _clean_content(s0)
        if CONTRA.match(s_snake):
            return s_snake
        m = re.match(r'(?i)^(no_evident_not|have_evident)[\s_:;.,-]*(.+)$', s0)
        if m:
            pref = m.group(1).lower()
            core = _clean_content(m.group(2))
            return f"{pref}_{core}" if core else ""
        return ""
    
    out = []
    lines = [ln.rstrip() for ln in txt.split('\n') if ln.strip()]
    i = 0
    last_support_token = None
    
    def consume_list(start_idx: int, is_contrary: bool) -> int:
        nonlocal last_support_token
        j = start_idx
        while j < len(lines):
            raw = lines[j].strip()
            if not raw:
                break
            nb = re.sub(r'\*\*(.*?)\*\*', r'\1', raw).strip()
            if re.match(r'(?i)^(supports?|contraries?|support|contrary)\s*[:：]?\s*$', nb):
                break
            t1 = re.sub(r'^\s*\d+\.\s*', '', raw)
            t2 = re.sub(r'^\s*[-*•]\s*', '', t1).strip()
            t2 = strip_parentheticals(t2)
            items = [x.strip() for x in re.split(r',\s*', re.sub(r'[;，；]\s*', ', ', t2)) if x.strip()]
            for item in items:
                if is_contrary:
                    contr_tok = normalize_contrary_payload(item)
                    if not contr_tok and last_support_token:
                        contr_tok = f"{expected_prefix}_{last_support_token}"
                    if contr_tok and contr_tok not in out:
                        out.append(contr_tok)
                else:
                    sup_tok = support_payload_to_token(item)
                    if sup_tok and sup_tok not in out:
                        out.append(sup_tok)
                        last_support_token = sup_tok
            j += 1
        return j
    
    while i < len(lines):
        line = lines[i]
        
        line_wo_num = re.sub(r'^\s*\d+\.\s*', '', line)
        line_wo_bul = re.sub(r'^\s*[-*•]\s*', '', line_wo_num)
        line_no_bold = re.sub(r'\*\*(.*?)\*\*', r'\1', line_wo_bul).strip()
        
        if re.match(r'(?i)^(here\s+is|based\s+on|assumption[-\s]*based|each\s+support|the\s+format|aba\s+format|final\s+response|this\s+supports?|this\s+format|structured\s+response)\b', line_no_bold):
            i += 1; continue
        if re.match(r'(?i)^claim\s*[:：]\s*', line_no_bold):
            i += 1; continue
        
        m_num_sup = re.match(r'^\d+\.\s*\*\*Support\*\*\s*:\s*(.+)$', line, re.I)
        if m_num_sup:
            sup_payload = m_num_sup.group(1).strip()
            sup_tok = support_payload_to_token(sup_payload)
            if sup_tok and sup_tok not in out:
                out.append(sup_tok)
                last_support_token = sup_tok
            
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                m_indent_con = re.match(r'^\s+\*\*Contrary\*\*\s*:\s*(.+)$', nxt, re.I)
                if m_indent_con:
                    contr_payload = m_indent_con.group(1).strip()
                    contr_tok = normalize_contrary_payload(contr_payload)
                    if not contr_tok and last_support_token:
                        contr_tok = f"{expected_prefix}_{last_support_token}"
                    if contr_tok and contr_tok not in out:
                        out.append(contr_tok)
                    i += 2
                    continue
            i += 1
            continue
        
        m_num_plain = re.match(r'^\d+\.\s+([a-z0-9_]+)$', line_no_bold, re.I)
        if m_num_plain:
            tok = m_num_plain.group(1).lower()
            if CONTRA.match(tok):
                if tok not in out and tok not in EXCLUDED_TOKENS:
                    out.append(tok)
            elif TOKEN.match(tok) and len(tok) >= 2 and tok not in EXCLUDED_TOKENS:
                if tok not in out:
                    out.append(tok)
                    last_support_token = tok
            i += 1
            continue
        
        m_num_arrow = re.match(r'^\d+\.\s+([a-z0-9_]+)\s*->\s*(.+)$', line_no_bold, re.I)
        if m_num_arrow:
            sup_payload = m_num_arrow.group(1).strip()
            contr_payload = m_num_arrow.group(2).strip()
            
            sup_tok = support_payload_to_token(sup_payload)
            if sup_tok and sup_tok not in out:
                out.append(sup_tok)
                last_support_token = sup_tok
            
            contr_tok = normalize_contrary_payload(contr_payload)
            if not contr_tok and last_support_token:
                contr_tok = f"{expected_prefix}_{last_support_token}"
            if contr_tok and contr_tok not in out:
                out.append(contr_tok)
            
            i += 1
            continue

        if ',' in line_no_bold and not re.search(r'(?i)\b(support|contrary)\b', line_no_bold):
            parts = [p.strip() for p in line_no_bold.split(',')]
            if len(parts) >= 2:
                first_part = parts[0].strip()
                if not re.match(r'(?i)^(no[_\s]+evident[_\s]+not|have[_\s]+evident)\b', first_part):
                    sup_tok = support_payload_to_token(first_part)
                    if sup_tok and sup_tok not in out:
                        out.append(sup_tok)
                        last_support_token = sup_tok

                    for part in parts[1:]:
                        part = part.strip()
                        if re.match(r'(?i)^(no[_\s]+evident[_\s]+not|have[_\s]+evident)\b', part):
                            contr_tok = normalize_contrary_payload(part)
                            if not contr_tok and last_support_token:
                                contr_tok = f"{expected_prefix}_{last_support_token}"
                            if contr_tok and contr_tok not in out:
                                out.append(contr_tok)
                        else:
                            sup_tok2 = support_payload_to_token(part)
                            if sup_tok2 and sup_tok2 not in out:
                                out.append(sup_tok2)
                    i += 1
                    continue
        
        m_sup = re.match(r'(?i)^\[?\s*\*?\*?support(?:s)?\*?\*?\s*\]?\s*[:：]\s*(.+)$', line_wo_bul)
        if m_sup:
            payload = m_sup.group(1).strip()
            parts = [p.strip() for p in re.split(r',\s*', re.sub(r'[;，；]\s*', ', ', payload)) if p.strip()]
            if parts:
                for part in parts:
                    sup_tok = support_payload_to_token(part)
                    if sup_tok and sup_tok not in out:
                        out.append(sup_tok); last_support_token = sup_tok
            else:
                sup_tok = support_payload_to_token(payload)
                if sup_tok and sup_tok not in out:
                    out.append(sup_tok); last_support_token = sup_tok
            i += 1; continue
        
        m_con = re.match(r'(?i)^\[?\s*\*?\*?contrary(?:ies)?\*?\*?\s*\]?\s*[:：]\s*(.+)$', line_wo_bul)
        if m_con:
            payload = m_con.group(1).strip()
            parts = [p.strip() for p in re.split(r',\s*', re.sub(r'[;，；]\s*', ', ', payload)) if p.strip()]
            if parts:
                for part in parts:
                    contr_tok = normalize_contrary_payload(part)
                    if not contr_tok and last_support_token:
                        contr_tok = f"{expected_prefix}_{last_support_token}"
                    if contr_tok and contr_tok not in out:
                        out.append(contr_tok)
            else:
                contr_tok = normalize_contrary_payload(payload)
                if not contr_tok and last_support_token:
                    contr_tok = f"{expected_prefix}_{last_support_token}"
                if contr_tok and contr_tok not in out:
                    out.append(contr_tok)
            i += 1; continue
        
        m_bold_sentence = re.match(r'^\*\*(.+?)\*\*\s*\.?$', line_wo_bul)
        if m_bold_sentence:
            sup_tok = support_payload_to_token(m_bold_sentence.group(1))
            if sup_tok and sup_tok not in out:
                out.append(sup_tok); last_support_token = sup_tok
            if i + 1 < len(lines):
                nxt = re.sub(r'\*\*(.*?)\*\*', r'\1', re.sub(r'^\s*[-*•]\s*', '', re.sub(r'^\s*\d+\.\s*', '', lines[i + 1]))).strip()
                m_con2 = re.match(r'(?i)^\[?\s*contrary(?:ies)?\s*\]?\s*[:：]\s*(.+)$', nxt)
                if m_con2:
                    contr_tok = normalize_contrary_payload(m_con2.group(1))
                    if not contr_tok and last_support_token:
                        contr_tok = f"{expected_prefix}_{last_support_token}"
                    if contr_tok and contr_tok not in out:
                        out.append(contr_tok)
                    i += 2; continue
            i += 1; continue
        
        m_hdr = re.match(r'(?i)^(supports?|contraries?)\s*[:：]\s*(.*)$', line_no_bold)
        if m_hdr:
            hdr = m_hdr.group(1).lower()
            payload = (m_hdr.group(2) or "").strip()
            if payload:
                items = [x.strip() for x in re.split(r',\s*', re.sub(r'[;，；]\s*', ', ', payload)) if x.strip()]
                if 'support' in hdr:
                    for item in items:
                        sup_tok = support_payload_to_token(item)
                        if sup_tok and sup_tok not in out:
                            out.append(sup_tok); last_support_token = sup_tok
                else:
                    for item in items:
                        contr_tok = normalize_contrary_payload(item)
                        if not contr_tok and last_support_token:
                            contr_tok = f"{expected_prefix}_{last_support_token}"
                        if contr_tok and contr_tok not in out:
                            out.append(contr_tok)
                i += 1; continue
            else:
                j = consume_list(i + 1, is_contrary=('contrar' in hdr))
                i = j; continue
        
        if not re.match(r'(?i)^\[?\s*contrary', line_no_bold) and not re.match(r'(?i)^(no[_\s]+evident[_\s]+not|have[_\s]+evident)\b', line_no_bold):
            sup_tok = None
            m_b = re.search(r'\*\*([a-z0-9_]+)\*\*', line_wo_bul, re.I)
            if m_b:
                sup_tok = m_b.group(1).lower()
            else:
                if "_" in line_no_bold:
                    m_sn = re.search(r'\b([a-z0-9]+(?:_[a-z0-9]+)*)\b', line_no_bold)
                    sup_tok = (m_sn.group(1).lower() if m_sn else "")
                else:
                    sup_tok = support_payload_to_token(line_no_bold)
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                nxt_wo = re.sub(r'\*\*(.*?)\*\*', r'\1', re.sub(r'^\s*[-*•]\s*', '', re.sub(r'^\s*\d+\.\s*', '', nxt))).strip()
                if re.match(r'(?i)^(no[_\s]+evident[_\s]+not|have[_\s]+evident)\b', nxt_wo) or re.match(r'(?i)^\[?\s*contrary(?:ies)?\s*\]?\s*[:：]', nxt_wo):
                    if sup_tok and TOKEN.match(sup_tok) and sup_tok not in out and sup_tok not in EXCLUDED_TOKENS:
                        out.append(sup_tok); last_support_token = sup_tok
                    if re.match(r'(?i)^\[?\s*contrary(?:ies)?\s*\]?\s*[:：]', nxt_wo):
                        contr_payload = re.sub(r'(?i)^\[?\s*contrary(?:ies)?\s*\]?\s*[:：]\s*', '', nxt_wo)
                        contr_tok = normalize_contrary_payload(contr_payload)
                    else:
                        contr_tok = normalize_contrary_payload(nxt_wo)
                    if not contr_tok:
                        contr_tok = f"{expected_prefix}_{(sup_tok or last_support_token or '').strip()}"
                    if contr_tok and contr_tok not in out:
                        out.append(contr_tok)
                    i += 2; continue
        
        lone_contra = normalize_contrary_payload(line_no_bold)
        if lone_contra:
            if lone_contra not in out:
                out.append(lone_contra)
            i += 1; continue
        
        if "_" in line_no_bold:
            m_sn2 = re.match(r'^([a-z0-9]+(?:_[a-z0-9]+)*)$', _clean_content(line_no_bold))
            if m_sn2:
                tok = m_sn2.group(1).lower()
                if TOKEN.match(tok) and tok not in out and tok not in EXCLUDED_TOKENS:
                    out.append(tok); last_support_token = tok
                i += 1; continue
        
        i += 1
    
    filtered, seen = [], set()
    for t in out:
        t = re.sub(r'\s+', '_', t.strip().strip('"\'* `')).lower()
        if (TOKEN.match(t) or CONTRA.match(t)) and t not in EXCLUDED_TOKENS:
            if len(t) > 120 or t.count('_') > 12:
                continue
            if t not in seen:
                seen.add(t); filtered.append(t)
    
    if not filtered:
        contraries = set(m.group(0).lower()
                         for m in re.finditer(r'\b(?:no_evident_not|have_evident)_[a-z0-9]+(?:_[a-z0-9]+)*\b', _clean_content(txt)))
        supports = set(m.group(0).lower()
                       for m in re.finditer(r'\b[a-z0-9]+(?:_[a-z0-9]+)+\b', _clean_content(txt))
                       if m.group(0).lower() not in EXCLUDED_TOKENS)
        fallback = []
        for s in supports:
            if s not in contraries and s not in fallback:
                fallback.append(s)
        for ctok in contraries:
            if ctok not in fallback:
                fallback.append(ctok)
        filtered = fallback
    
    return " , ".join(filtered)


def _merge_chunk_into_excel(chunk_df: pd.DataFrame, test_col: str, sheet_path: str, cols_all: list):
    for c in ["ID", "Prompt", "Topic", test_col]:
        if c not in chunk_df.columns:
            chunk_df[c] = ""
    if os.path.exists(sheet_path):
        prev = pd.read_excel(sheet_path)
        for c in cols_all:
            if c not in prev.columns: prev[c] = ""
        for c in prev.columns:
            if c not in chunk_df.columns: chunk_df[c] = ""
        prev_ids = {str(rid): i for i, rid in enumerate(prev["ID"].astype(str).tolist())} if "ID" in prev.columns else {}
        for _, r in chunk_df.iterrows():
            rid = str(r["ID"])
            if rid in prev_ids:
                i = prev_ids[rid]
                prev.at[i, "Prompt"] = r["Prompt"]
                prev.at[i, "Topic"]  = r["Topic"]
                prev.at[i, test_col] = r[test_col]
            else:
                prev = pd.concat([prev, r.to_frame().T], ignore_index=True)
        prev = prev.drop_duplicates(subset=["ID"], keep="last")
        prev.to_excel(sheet_path, index=False)
    else:
        base = pd.DataFrame(columns=cols_all)
        for c in ["ID", "Prompt", "Topic", test_col]:
            base[c] = chunk_df[c]
        base.to_excel(sheet_path, index=False)

def _ids_from_log(log_path: str):
    processed = set()
    if not os.path.exists(log_path):
        return processed
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ID="):
                    val = line.split("=", 1)[1].strip()
                    try:
                        processed.add(int(val)) if val.isdigit() else processed.add(val)
                    except Exception:
                        processed.add(val)
    except Exception:
        pass
    return processed

df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]
required = {TEXT_COL, TOPIC_COL, POSNEG_COL, ID_COL}
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in Excel: {missing}. Found: {list(df.columns)}")

sel, sample_set = [], set(SAMPLE_IDS) if SAMPLE_IDS else None
for idx, r in df.iterrows():
    text, topic, posneg = r.get(TEXT_COL, ""), r.get(TOPIC_COL, ""), r.get(POSNEG_COL, "")
    rid = coerce_id(r.get(ID_COL, None))
    if not isinstance(text, str) or not text.strip(): continue
    if not topic_matches(topic, TARGET_TOPIC): continue
    pol = polarity(posneg)
    if pol is None: continue
    if rid in (None, "") or (isinstance(rid, float) and pd.isna(rid)): rid = idx + 1
    if sample_set is not None and (rid not in sample_set): continue
    claim = f"{pol}_{snake(TARGET_TOPIC)}"
    sel.append({"ID": rid, "Text": text.strip(), "Claim": claim})

df_sel = pd.DataFrame(sel).reset_index(drop=True)
if (sample_set is None) and isinstance(LIMIT_ROWS, int):
    df_sel = df_sel.head(LIMIT_ROWS).reset_index(drop=True)

if df_sel.empty:
    raise SystemExit("No matching rows after filtering/sampling.")

topic_snake = snake(TARGET_TOPIC)
out_dir = f"ABA_task2_{topic_snake}_{SAFE_TAG}"
os.makedirs(out_dir, exist_ok=True)

run_base = f"{topic_snake}_{SAFE_TAG}"      
sheet_path = os.path.join(out_dir, f"{run_base}_sheet.xlsx")
cols_all = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, N_RUNS + 1)] + ["Label"]

all_runs_tests = []
for run in range(1, N_RUNS + 1):
    test_col = f"Test {run}"
    run_csv  = os.path.join(out_dir, f"{run_base}_run{run}.csv")
    log_path = os.path.join(out_dir, f"log_run_{run}.txt")
    
    processed_ids = _ids_from_log(log_path)
    rows_this_session = []
    processed_this_session = 0
    
    if PRINT_TO_CONSOLE:
        print(f"\n========== RUN {run} / {N_RUNS} (resume from {len(processed_ids)} IDs) ==========")
    
    with open(log_path, "a", encoding="utf-8") as logf:
        for _, rr in df_sel.iterrows():
            rid = rr["ID"]
            if rid in processed_ids:
                continue
            
            msgs = build_messages(rr["Text"], rr["Claim"])
            prm  = stringify_messages(msgs)
            ans_raw = ask_llm_messages(msgs)
            
            if not ans_raw:
                ans_raw = "[ERROR] No response from model. Skipping."
            
            ans_out = postformat_aba(ans_raw, rr["Claim"]) if USE_POSTFORMAT else ans_raw
            
            raw_text = (ans_raw or "").replace("\r\n", "\n").rstrip()
            logf.write(f"ID={rid}\n")
            logf.write(raw_text + "\n\n")
            logf.flush()
            
            rows_this_session.append({
                "ID": rid,
                "Prompt": prm,
                "Topic": rr["Claim"],
                test_col: ans_out
            })
            processed_ids.add(rid)
            processed_this_session += 1
            
            if PRINT_TO_CONSOLE:
                print(f"[Run{run}] ID={rid}\n{ans_out}\n")
            
            if processed_this_session % CHUNK_SIZE == 0:
                part_df = pd.DataFrame(rows_this_session)
                if os.path.exists(run_csv):
                    try:
                        prev = pd.read_csv(run_csv)
                        merged = pd.concat([prev, part_df], ignore_index=True)
                        merged = merged.drop_duplicates(subset=["ID"], keep="last")
                        merged.to_csv(run_csv, index=False)
                    except Exception:
                        part_df.to_csv(run_csv, index=False)
                else:
                    part_df.to_csv(run_csv, index=False)
                
                _merge_chunk_into_excel(
                    part_df[["ID", "Prompt", "Topic", test_col]].copy(),
                    test_col, sheet_path, cols_all
                )
                
                if PRINT_TO_CONSOLE:
                    print(f"[Run {run}] Checkpoint: +{CHUNK_SIZE} rows saved to CSV & Excel.")
                
                rows_this_session = []
            
            time.sleep(0.20)
    
    if rows_this_session:
        part_df = pd.DataFrame(rows_this_session)
        if os.path.exists(run_csv):
            try:
                prev = pd.read_csv(run_csv)
                merged = pd.concat([prev, part_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["ID"], keep="last")
                merged.to_csv(run_csv, index=False)
            except Exception:
                part_df.to_csv(run_csv, index=False)
        else:
            part_df.to_csv(run_csv, index=False)
        
        _merge_chunk_into_excel(
            part_df[["ID", "Prompt", "Topic", test_col]].copy(),
            test_col, sheet_path, cols_all
        )
    
    try:
        final_run_df = pd.read_csv(run_csv)
    except Exception:
        final_run_df = pd.DataFrame(columns=["ID", "Prompt", "Topic", test_col])
    
    all_runs_tests.append(final_run_df[[test_col]] if test_col in final_run_df.columns else pd.DataFrame({test_col: []}))
    
    print(f"[Run {run}] wrote RAW log: {log_path}")
    print(f"[Run {run}] updated per-run CSV: {run_csv}")

wide = pd.DataFrame({
    "ID": df_sel["ID"],
    "Prompt": [stringify_messages(build_messages(t, c)) for t, c in zip(df_sel["Text"], df_sel["Claim"])],
    "Topic": df_sel["Claim"],
})

for run in range(1, N_RUNS + 1):
    if len(all_runs_tests) >= run and not all_runs_tests[run-1].empty:
        wide = pd.concat([wide, all_runs_tests[run-1]], axis=1)

wide["Label"] = ""
cols = ["ID", "Prompt", "Topic"] + [f"Test {i}" for i in range(1, N_RUNS + 1)] + ["Label"]
for c in cols:
    if c not in wide.columns:
        wide[c] = ""

wide = wide[cols].drop_duplicates(subset=["ID"], keep="last")

if os.path.exists(sheet_path):
    prev = pd.read_excel(sheet_path)
    for c in cols:
        if c not in prev.columns: prev[c] = ""
    for c in prev.columns:
        if c not in wide.columns: wide[c] = ""
    
    prev_ids = {str(rid): i for i, rid in enumerate(prev["ID"].astype(str).tolist())} if "ID" in prev.columns else {}
    for _, r in wide.iterrows():
        rid = str(r["ID"])
        if rid in prev_ids:
            i = prev_ids[rid]
            for c in cols:
                val = r.get(c, "")
                if pd.notna(val) and str(val) != "":
                    prev.at[i, c] = val
        else:
            prev = pd.concat([prev, r.to_frame().T], ignore_index=True)
    
    prev = prev[cols].drop_duplicates(subset=["ID"], keep="last")
    prev.to_excel(sheet_path, index=False)
else:
    wide.to_excel(sheet_path, index=False)

print(f"\nSaved per-run CSVs to: {out_dir}")
print(f"Appended/created sheet: {sheet_path}")