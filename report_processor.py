import re
from recist_eval import eval_recist_result
from datetime import datetime
import os
import json
from tools.lightrag import query_lightrag
from utils.tokentracker import estimate_tokens

role_desc = "You are an experienced breast tumor radiologist with expertise in stepwise reasoning and structured output."
with open("template/lesion_list_template.json", "r", encoding="utf-8") as f:
    lesion_template = json.load(f)
lesion_list_template_str = json.dumps(lesion_template["2.4 Lesion List"], ensure_ascii=False, indent=2)
KEY_FIELDS = ["current_lesion_list", "Treatment Efficacy Result", "Radiological Diagnosis"]
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

def query_lightrag_with_token(prompt, desc, model="gpt-4o"):
    result = query_lightrag(prompt, desc)
    prompt_token = estimate_tokens(prompt, model=model)
    result_token = estimate_tokens(str(result), model=model)
    total_token = prompt_token + result_token
    return result, total_token

def agent_lesion_list(memory, shot_mode=0):
    history_text = "\n\n====Historical Follow-up Description Split====\n\n".join(memory.get("history_descriptions", []))
    few_shot = load_few_shot(shot_mode, agent_name="lesion_list")
    history_lesion_lists = memory.get('history_lesion_lists', [])
    history_lesion_lists_str = (
        json.dumps(history_lesion_lists, ensure_ascii=False, indent=2) if history_lesion_lists else 'None'
    )

    kb_query = "Summarize the key points for distinguishing MRI breast lesions"
    kb_context = "Knowledge points summary, excluding patient cases."
    kb_points, kb_token = query_lightrag_with_token(kb_query, kb_context)
    phase = "Baseline" if is_baseline_case(memory) else "Follow-up"
    prompt = (
        "*** Please reference the supplementary materials. ***"
    )

    result, token = query_lightrag_with_token(prompt, memory.get("current_description", ""))
    res = (result or "").strip()

    if is_baseline_case(memory):
        try:
            js = json.loads(extract_json_from_llm_output(res))
            if lesion_list_contains_birads6(js):
                print("[Guard] Baseline case detected BI-RADS:6 in lesion list → Clearing and retrying")
                return "", token
        except Exception:
            if re.search(r"BI\s*[-–—]?\s*RADS\s*[:：]?\s*6\b", res, flags=re.I):
                print("[Guard] Suspected BI-RADS:6 in baseline case text → Clearing and retrying")
                return "", token
    if not is_baseline_case(memory):
        try:
            js = json.loads(extract_json_from_llm_output(res))
            if not lesion_list_contains_birads6(js):
                print("[Guard] Follow-up case lesion list does not include BI-RADS:6 → Clearing and retrying")
                return "", token
        except Exception:
            if not re.search(r"BI\s*[-–—]?\s*RADS\s*[:：]?\s*6\b", res, flags=re.I):
                print("[Guard] BI-RADS:6 not found in follow-up case text → Clearing and retrying")
                return "", token

    return res, token


def eval_response_agent(memory, shot_mode=0):
    history_desc = memory.get("history_descriptions", []) or []
    history_text = "\n\n====Historical Follow-up Description Split====\n\n".join(map(str, history_desc))
    history_lesion_lists = memory.get("history_lesion_lists", []) or []
    history_lesion_lists_str = (
        json.dumps(history_lesion_lists, ensure_ascii=False, indent=2) if history_lesion_lists else "None"
    )
    current_desc = memory.get("current_description", "") or ""
    current_lesion_list = (
        memory.get("lesion_list", []) or memory.get("current_lesion_list", []) or []
    )
    if isinstance(current_lesion_list, dict) and "lesion_list" in current_lesion_list:
        current_lesion_list = current_lesion_list["lesion_list"]
    current_lesion_list_str = json.dumps(current_lesion_list, ensure_ascii=False, indent=2)
    few_shots = load_few_shot(shot_mode, agent_name="recist")
    label = eval_recist_result(
        history_text=history_text,
        current_desc=current_desc,
        history_lesion_lists_str=history_lesion_lists_str,
        current_lesion_list_str=current_lesion_list_str,
        few_shot=few_shots,
    )
    return (label or "").strip(), 0

def agent_radiology_diag(current_desc, memory, shot_mode=0):
    if not current_desc:
        return "", 0

    few_shot = load_few_shot(shot_mode, agent_name="diag")
    history_lesion_lists = memory.get('history_lesion_lists', [])
    history_lesion_lists_str = (
        json.dumps(history_lesion_lists, ensure_ascii=False, indent=2) if history_lesion_lists else 'None'
    )
    current_lesion_list = memory.get('current_lesion_list', [])
    history_desc = memory.get("history_descriptions", [])
    history_text = "\n\n====Historical Follow-up Description Split====\n\n".join(history_desc) if history_desc else "None"
    NEED_BIRADS_TYPES = {"Mass", "Non-mass Enhancement", "Mass with Non-mass", "Punctate Enhancement", "Ductal Dilatation", "Structural Distortion"}

    def normalize_birads(x: str) -> str:
        s = str(x or "").upper()
        s = s.replace("BI-RADS", "").replace("BIRADS", "")
        s = re.sub(r'[\s:：.,.，；;]+', '', s)
        return s  # Expect to return 1/2/3/4A/4B/4C/5/6

    def build_core_text_from_current(current_lesion_list):
        specs = []
        for it in (current_lesion_list or []):
            typ = (it.get("Lesion Type") or "").strip()
            if typ not in NEED_BIRADS_TYPES:
                continue
            loc = it.get("2.4.1 Location") or {}
            side = (loc.get("2.4.1.1 Side") or "").strip()
            quad = (loc.get("2.4.1.2 Quadrant") or "").strip()
            depth = (loc.get("2.4.1.3 Depth") or "").strip()
            bir_raw = (
                    ((it.get("2.4.5 BI-RADS Classification") or {}).get("2.4.5.1 BI-RADS"))
                    or it.get("BI-RADS Rating") or it.get("BI_RADS") or ""
            )
            bir = normalize_birads(bir_raw)
            if bir not in {"1", "2", "3", "4A", "4B", "4C", "5", "6"}:
                continue
            specs.append({"side": side, "quad": quad, "depth": depth, "type": typ, "birads": bir})
        lines = []
        for s in specs:
            pos = "".join([s["side"], s["quad"], s["depth"]])
            line = f"{pos}{s['type']}，BI-RADS:{s['birads']}."
            lines.append(line)
        core_text = "".join(lines)
        return core_text, specs

    core_text, specs = build_core_text_from_current(memory.get("current_lesion_list", []))
    print(f"[Generation] Total {len(specs)} lesions that need to be graded:", [(s['type'], s['birads']) for s in specs])
    print(core_text)
    N = len(specs)
    tok1 = 0
    if N > 0:
        refine_prompt = (
            "*** Please reference the supplementary materials. ***"
        )

        def tidy_core_text(text: str) -> str:
            t = (text or "").replace("\r", "")
            lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
            lines = [ln if re.search(r'[..!？?]$', ln) else (ln + ".") for ln in lines]
            lines = [re.sub(r'[.]{2,}$', '.', ln) for ln in lines]
            return "".join(lines)

        core_text_raw, tok1 = query_lightrag_with_token(refine_prompt, current_desc)
        core_text = tidy_core_text(core_text_raw)
        print(core_text)
        core_text = re.sub(r"\b\d+(\.\d+)?\s*(mm|cm|%)\b|约\s*\d+(\.\d+)?\s*(mm|cm)|[×xX*]", "", core_text)
        sents = [s for s in re.split(r"[.!?？！]", core_text) if s.strip()]
        PATTERN = r"""
        BI\s*[-–—]?\s*RADS           
        (?:\s*(?:Classification|Grade))?        
        \s*[:：]?\s*
        (?:
            (?<!\d)4\s*[ABC](?![A-Z])  
          | (?<!\d)[02356](?!\d)     
        )
        (?:\s*[grade|type])?            
        """

        lesion_sents = [s for s in sents if re.search(PATTERN, s, flags=re.I | re.X)]
        lesion_cnt = len(lesion_sents)
        expected_cnt = len(specs)
        if lesion_cnt != expected_cnt:
            print(f"[Guard] The number of lesion sentences does not match: Expected {expected_cnt} sentences, but found {lesion_cnt} → Prompt retry.")
            core_text = ""

    else:
        tok1 = 0
    extra_prompt = (
        "*** Please reference the supplementary materials. ***"
    )
    extra_result, tok2 = query_lightrag_with_token(extra_prompt, current_desc)
    extra_text = (extra_result or "").strip().replace("\r", "").replace("\n", "")
    if "." in extra_text:
        extra_text = extra_text.split(".")[0] + "."
    extra_text = re.sub(r"BI-RADS\s*:[^.]*", "", extra_text, flags=re.IGNORECASE)
    for pat in [r"\b\d+(\.\d+)?\s*(mm|cm)\b", r"[×xX*]", r"约\s*\d+(\.\d+)?\s*(mm|cm)",
                r"\b(T1|T2|DWI|ADC|MIP)\b", r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", r"\d{4}年\d{1,2}月\d{1,2}日"]:
        extra_text = re.sub(pat, "", extra_text, flags=re.IGNORECASE)
    extra_text = re.sub(r"(Skin | Subcutaneous tissue | Edema | Thickening)[^，.]*", "", extra_text)
    extra_text = extra_text.replace("、", "，").strip()
    if extra_text.endswith("."):
        tmp = extra_text[:-1]
    else:
        tmp = extra_text

    clauses = [c.strip() for c in tmp.split("，") if c.strip()]
    ph_re = r"(Consider metastasis|Invasion|Please correlate with other examinations)"
    idx = None
    for i, c in enumerate(clauses):
        if re.search(ph_re, c):
            idx = i
            break

    if idx is not None:
        keep = [clauses[idx - 1], clauses[idx]] if idx > 0 else [clauses[idx]]
    else:
        keep = clauses[:2]
    _keep2 = []
    for c in keep:
        c2 = re.sub(r"(Skin | Subcutaneous tissue | Edema | Thickening)[^，.]*", "", c).strip()
        if c2:
            _keep2.append(c2)
    keep = _keep2
    extra_text = "，".join(keep) + ("." if keep else "")
    extra_text = re.sub(r"[，、]+\s*.", ".", extra_text)
    extra_text = re.sub(r"，{2,}", "，", extra_text).strip()
    has_site = re.search(r"(Axilla|Armpit|Lymph node|Internal mammary|Supraclavicular|Infraclavicular|Nipple|Areola|Chest wall|Pectoral muscle|Bone|Pleura)", extra_text)
    phrases = re.findall(ph_re, extra_text)
    only_verb = extra_text in {"Invasion.", "Consider metastasis.", "Please correlate with other examinations."}
    if (not extra_text) or (not has_site) or only_verb or len(phrases) != 1:
        extra_text = ""
    core_out = core_text.strip()
    if core_out == "":
        core_out = "#"
    final_text = f"{core_out}{extra_text}"
    if is_baseline_case(memory) and re.search(r"BI\s*[-–—]?\s*RADS\s*[:：]?\s*6\b", final_text, flags=re.I):
        print("[Guard] Baseline case detected BI-RADS:6 in radiology diagnosis → Clearing and retrying")
        final_text = ""
    if not is_baseline_case(memory) and not re.search(r"BI\s*[-–—]?\s*RADS\s*[:：]?\s*6\b", final_text, flags=re.I):
        print("[Guard] Follow-up case does not include BI-RADS:6 in radiology diagnosis → Clearing and retrying")
        final_text = ""
    if contains_dirty_tokens(final_text):
        print("[Guard] The diagnosis text contains suspected references/links/code or dirty tokens → Clearing for retry")
        final_text = ""
    final_text = _dedup_keep_one(final_text)
    return final_text, tok1 + tok2

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

        if not memory.get("Treatment Efficacy Result") and memory.get("current_lesion_list"):
            print("[MemoryChecker] Processing efficacy evaluation...")
            try:
                eval_result, eval_token = eval_response_agent(memory, shot_mode=shot_mode)
                total_tokens += eval_token
                if eval_result:
                    memory["Treatment Efficacy Result"] = eval_result
                    if not all_eval_result_have(eval_result):
                        print("Stored content in memory:", memory["Treatment Efficacy Result"])
                        print("[MemoryChecker] Detected invalid efficacy evaluation result, clearing and retrying next round.")
                        memory["Treatment Efficacy Result"] = ""
                        updated = True
                    else:
                        print("Stored content in memory:", memory["Treatment Efficacy Result"])
                        updated = True
            except Exception as e:
                print(f"[MemoryChecker] Efficacy evaluation request failed, skipping this round: {e}")
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


