import re
from datetime import datetime
import os
import json
from utils.chat import call_gpt_with_token

with open("template/lesion_list_template.json", "r", encoding="utf-8") as f:
    lesion_template = json.load(f)
lesion_list_template_str = json.dumps(lesion_template["2.4 Lesion List"], ensure_ascii=False, indent=2)
KEY_FIELDS = ["current_lesion_list", "Radiological Diagnosis"]
FORBIDDEN_PATTERNS = [
    r"(?i)\.pdf\b",
    r"(?i)\bpdf\b",
    r"https?://",
    r"\[[^\]]*?\]",
    r"`{3,}",
    r"[{}]",
]
def _dedup_keep_one(final_text: str) -> str:
    sents = [s.strip() for s in re.split(r"[.\.]", final_text or "") if s.strip()]
    seen, out = set(), []
    removed = []

    for s in sents:
        norm = re.sub(r"\s+", "", s)
        if norm in seen:
            removed.append(s)
            continue
        seen.add(norm)
        out.append(s)


    if removed:
        uniq = []
        seen_norm = set()
        for s in removed:
            n = re.sub(r"\s+", "", s)
            if n not in seen_norm:
                seen_norm.add(n)
                uniq.append(s)
        print(f"[Guard] Duplicate lesions detected, merged:{'；'.join(uniq[:3])}")

    return ".".join(out) + ("." if out else "")
def _append_failure_line(missing, case_id=None, rounds_done=None, max_rounds=None,
                         out_path="structured_reports/all_failures.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rec = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_id": case_id or "unknown",
        "missing": missing,
        "rounds_done": rounds_done,
        "max_rounds": max_rounds,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def contains_dirty_tokens(text: str) -> bool:
    t = text or ""
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def is_baseline_case(memory: dict) -> bool:
    if "_is_baseline" in memory and memory["_is_baseline"] is not None:
        return bool(memory["_is_baseline"])
    return not (memory.get("history_descriptions") or memory.get("history_lesion_lists"))

def normalize_birads_value(x: str) -> str:
    s = str(x or "").upper()
    s = s.replace("BI-RADS", "").replace("BIRADS", "")
    s = re.sub(r'[\s:：.,.，；;]+', '', s)
    return s

def lesion_list_contains_birads6(lesion_list) -> bool:
    try:
        for it in (lesion_list or []):
            raw = ((it.get("2.4.5 BI-RADS Classification") or {}).get("2.4.5.1 BI-RADS", "")) or it.get("BI-RADS Grading") or it.get("BI_RADS") or ""
            if normalize_birads_value(raw) == "6":
                return True
    except Exception:
        pass
    return False

_BLOCK_PATTERNS = [
    r'#', r'\breference\b',
    r'\bT1\b', r'\bT2\b', r'\bDWI\b', r'\bADC\b', r'\bMIP\b',
    r'\s*\d+(?:\.\d+)?\s*(?:cm|mm)',
    r'\d+(?:\.\d+)?\s*(?:cm|mm)',
    r'\d+\s*[×xX*]\s*\d+',
]

def all_diag_text_have(x) -> bool:
    t = str(x or "").strip()
    if not t:
        return False
    u = t.upper().replace("×", "x")
    for pat in _BLOCK_PATTERNS:
        if re.search(pat, u, flags=re.IGNORECASE):
            return False
    return True

def all_eval_result_have(x):
    return str(x).strip() in {"baseline", "CR", "PR", "SD", "PD"}
def all_lesions_have_birads(lesion_list):
    if not isinstance(lesion_list, list) or not lesion_list:
        return False
    for lesion in lesion_list:
        try:
            val = lesion.get("2.4.5 BI-RADS Classification", {}).get("2.4.5.1 BI-RADS", "")
            if not isinstance(val, str) or not val.strip():
                return False
        except Exception:
            return False
    return True
def extract_json_from_llm_output(llm_output):
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, llm_output)
    if match:
        json_str = match.group(1)
    else:
        json_str = llm_output.strip()
    return json_str
def load_few_shot(shot_mode=0, agent_name="birads"):
    fname_map = {
        "birads": "shots/birads.json",
        "diag": "shots/diag.json",
        "recist": "shots/recist.json",
        "lesion_list": "shots/lesion_list.json",    # 新增一行
    }
    fname = fname_map.get(agent_name)
    if not fname or not os.path.exists(fname):
        return ""
    with open(fname, encoding="utf-8") as f:
        data = json.load(f)
    if shot_mode == 1:
        exs = data.get("oneshot", [])
    elif shot_mode == 2:
        exs = data.get("fewshot", [])
    else:
        return data.get("zeroshot", "")
    example_strs = []
    for item in exs:
        q = item.get("question", "")
        a = item.get("answer", "")
        example_strs.append(f"Example:\nDescription:{q}\nOutput: {a}")
    return "\n\n".join(example_strs) + "\n\n" if example_strs else ""

def clean_radiology_diag(text):
    text = re.sub(r'\n*#+\s*[^\n]+\n*', '\n', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^[\s\-•\d]+\.(?=\s)', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.strip()
    return text


def agent_lesion_list(memory, shot_mode=0):
    history_text = "\n\n====Historical Follow-up Description Split====\n\n".join(memory.get("history_descriptions", []))
    few_shot = load_few_shot(shot_mode, agent_name="lesion_list")
    history_lesion_lists = memory.get('history_lesion_lists', [])
    history_lesion_lists_str = (
        json.dumps(history_lesion_lists, ensure_ascii=False, indent=2) if history_lesion_lists else 'None'
    )
    current_description = memory.get("current_description", "") or ""
    phase = "Baseline" if is_baseline_case(memory) else "Follow-up"
    with open("template/lesion_list_template.json", "r", encoding="utf-8") as f:
        lesion_template = json.load(f)
    lesion_list_JSON_template = json.dumps(lesion_template["2.4 Lesion List"], ensure_ascii=False, indent=2)
    prompt = (
        "*** Please reference the supplementary materials. ***"
    )
    result, token = call_gpt_with_token(prompt)
    res = (result or "").strip()
    return res, token


def agent_radiology_diag(current_desc, memory, shot_mode=0):
    historical_description = "\n".join(memory.get("history_descriptions", []))
    history_lesion_lists = json.dumps(memory.get("history_lesion_lists", []), ensure_ascii=False, indent=2)
    historical_impression = "\n".join(memory.get("history_impressions", []))
    current_description = memory.get("current_description", "") or ""
    current_lesion_list = json.dumps(memory.get("current_lesion_list", []), ensure_ascii=False, indent=2)
    one_shot_radiology_impression_JSON_example = load_few_shot(shot_mode, agent_name="diag")
    with open("template/structured_report_template.json", encoding="utf-8") as f:
        structured_report = json.load(f)
    radiology_impression_JSON_template = json.dumps(structured_report["3 Radiological Diagnosis"], ensure_ascii=False, indent=2)
    prompt = (
        "*** Please reference the supplementary materials. ***"
    )
    result, token = call_gpt_with_token(prompt)
    res = (result or "").strip()
    return res, token


def memory_checker(memory, is_baseline_flag=None, max_rounds=10, shot_mode=0):
    total_tokens = 0
    round_idx = 0
    if is_baseline_flag is not None:
        memory["_is_baseline"] = bool(is_baseline_flag)
    else:
        memory["_is_baseline"] = None
    while round_idx < max_rounds:
        round_idx += 1
        print(f"\n[MemoryChecker] Running round {round_idx} of inference...")
        updated = False
        if not memory.get("current_lesion_list"):
            print("[MemoryChecker] Extracting lesion list and grading...")
            try:
                lesion_list_str, token = agent_lesion_list(memory, shot_mode=shot_mode)
                total_tokens += token
                if lesion_list_str:
                    lesion_list_str = extract_json_from_llm_output(lesion_list_str)
                    lesion_list = json.loads(lesion_list_str)
                    memory["current_lesion_list"] = lesion_list
                    if not all_lesions_have_birads(lesion_list):
                        print("Stored content in memory:", memory["current_lesion_list"])
                        print("[MemoryChecker] Detected missing BI-RADS in lesion list, clearing result and retrying next round.")
                        memory["current_lesion_list"] = []
                        updated = True
                    else:
                        print("Stored content in memory:", memory["current_lesion_list"])
                        updated = True
            except Exception as e:
                print(f"[MemoryChecker] Lesion list parsing or request failed, skipping this round: {e}")
                continue

        if not memory.get("Radiological Diagnosis") and memory.get("current_lesion_list"):
            print("[MemoryChecker] Processing radiological diagnosis...")
            try:
                diag_text, diag_token = agent_radiology_diag(
                    memory.get("current_description", ""), memory, shot_mode=shot_mode
                )
                total_tokens += diag_token
                if diag_text:
                    cleaned = clean_radiology_diag(diag_text)
                    memory["Radiological Diagnosis"] = clean_radiology_diag(diag_text)
                    cc = memory.get("current_lesion_list", [])
                    if not all_diag_text_have(cleaned):
                        print("Stored content in memory:", memory["Radiological Diagnosis"])
                        print("[MemoryChecker] Radiological diagnosis does not match lesion grading or contains forbidden words, clearing and retrying.")
                        memory["Radiological Diagnosis"] = ""
                        updated = True
                    else:
                        print("Stored content in memory:", memory["Radiological Diagnosis"])
                        updated = True
            except Exception as e:
                print(f"[MemoryChecker] Radiological diagnosis request failed, skipping this round: {e}")
                continue

        if all(memory.get(f, "") for f in KEY_FIELDS):
            print(f"[MemoryChecker] All key fields have been generated in round {round_idx}, ending early.")
            break

    print("[MemoryChecker] Inference ended.")

    if round_idx >= max_rounds:
        missing = [k for k in KEY_FIELDS if not memory.get(k)]
        if missing:
            case_id = memory.get("patient_id") or memory.get("case_id") or memory.get("Patient ID")
            _append_failure_line(missing, case_id=case_id, rounds_done=round_idx, max_rounds=max_rounds)

    return memory, total_tokens






