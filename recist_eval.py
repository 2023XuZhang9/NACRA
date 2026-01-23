# -*- coding: utf-8 -*-
import os
import re
import json
import datetime
import requests
from typing import Any, Dict, List, Optional, Tuple

LIGHTRAG_URL    = os.getenv("LIGHTRAG_URL","xxx")
REQUEST_TIMEOUT = int(os.getenv("LR_TIMEOUT", "120"))
MAX_ATTEMPTS    = 10
DEBUG_PRINT     = bool(int(os.getenv("LR_DEBUG", "1")))
ALLOWED_RECIST  = {"CR", "PR", "SD", "PD", "Baseline"}
DECISION_LOG_PATH = os.getenv(
    "DECISION_LOG_PATH",
    os.path.join("structured_reports", "recist_eval_log.jsonl")
)

def _log_decision(label: str,
                  method: str,
                  meta: Optional[Dict[str, Any]] = None,
                  trace_id: Optional[str] = None) -> None:
    try:
        os.makedirs(os.path.dirname(DECISION_LOG_PATH), exist_ok=True)
        rec: Dict[str, Any] = {
            "ts": datetime.datetime.now().astimezone().isoformat(),
            "trace_id": trace_id,
            "method": method,
            "label": label
        }
        if meta:
            rec["details"] = meta
        with open(DECISION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        if DEBUG_PRINT:
            print(f"[DecisionLog] fail：{e}")
def recist_calculator(
    baseline_sld_mm: Optional[float] = None,
    current_sld_mm: Optional[float] = None,
    nadir_sld_mm: Optional[float] = None,
    new_lesions: bool = False,
    non_target_pd: bool = False
) -> Dict[str, Any]:
    if current_sld_mm is None:
        return {"ok": False, "error": "缺少 current_sld_mm"}

    def _f(x):
        return None if x in (None, "") else float(x)

    cur = float(current_sld_mm)
    base = _f(baseline_sld_mm)
    nadir = _f(nadir_sld_mm)

    if new_lesions or non_target_pd:
        return {
            "ok": True,
            "baseline_sld_mm": base, "nadir_sld_mm": nadir, "current_sld_mm": cur,
            "delta_vs_baseline_pct": None if not base else (cur - base) / base * 100.0,
            "delta_vs_nadir_pct":    None if not nadir else (cur - nadir) / nadir * 100.0,
        }

    if cur == 0.0:
        return {
            "ok": True, 
            "baseline_sld_mm": base, "nadir_sld_mm": nadir, "current_sld_mm": cur,
            "delta_vs_baseline_pct": None if not base else -100.0,
            "delta_vs_nadir_pct":    None if not nadir else (cur - nadir) / nadir * 100.0,
        }

    if nadir not in (None, 0.0):
        rel = (cur - nadir) / nadir * 100.0
        abs_inc = cur - nadir
        if rel >= 20.0 and abs_inc >= 5.0:
            return {
                "ok": True, 
                "baseline_sld_mm": base, "nadir_sld_mm": nadir, "current_sld_mm": cur,
                "delta_vs_baseline_pct": None if not base else (cur - base) / base * 100.0,
                "delta_vs_nadir_pct": rel,
            }

    if base not in (None, 0.0):
        rel = (cur - base) / base * 100.0
        if rel <= -30.0:
            return {
                "ok": True, 
                "baseline_sld_mm": base, "nadir_sld_mm": nadir, "current_sld_mm": cur,
                "delta_vs_baseline_pct": rel,
                "delta_vs_nadir_pct": None if not nadir else (cur - nadir) / nadir * 100.0,
            }

    return {
        "ok": True,
        "baseline_sld_mm": base, "nadir_sld_mm": nadir, "current_sld_mm": cur,
        "delta_vs_baseline_pct": None if not base else (cur - base) / base * 100.0,
        "delta_vs_nadir_pct":    None if not nadir else (cur - nadir) / nadir * 100.0,
    }

def lightrag_query(user_prompt: str, query_title: str) -> str:
    payload = {"query": query_title, "mode": "local", "user_prompt": user_prompt}
    r = requests.post(LIGHTRAG_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    text = data.get("answer") or data.get("response") or data.get("content") or json.dumps(data, ensure_ascii=False)
    if DEBUG_PRINT:
        preview = (str(text)[:200] + " …") if len(str(text)) > 200 else str(text)
        print("  [agent determination Result]:", preview)
    return text

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.IGNORECASE)

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(str(text))
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _looks_like_empty_json_str(s: str) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if t in ("", "[]", "{}", "null", "None"):
        return True
    try:
        obj = json.loads(t)
        if isinstance(obj, (list, dict)) and len(obj) == 0:
            return True
    except Exception:
        pass
    return False

def _is_baseline_first_visit(history_text: str, history_lesion_lists_str: str) -> bool:
    ht = (history_text or "").strip()
    if ht:
        return False
    return _looks_like_empty_json_str(history_lesion_lists_str)


def _recist_via_lightrag_tool(history_text: str,
                               current_desc: str,
                               history_lesion_lists_str: str,
                               current_lesion_list_str: str,
                               max_attempts: int = MAX_ATTEMPTS,
                               few_shot: str = "") -> Optional[Tuple[str, Dict[str, Any]]]:
    examples_block = f"\n【Sample Results for Small Sample Size (For Reference Only)】\n{few_shot}\n" if (few_shot and str(few_shot).strip()) else ""
    system_text = (
        "*** Please reference the supplementary materials. ***"
    )

    base_user = (
        f"Historical Description:\n{history_text}\n"
        f"Current Description:\n{current_desc}\n"
        f"Historical Structured Lesion List:\n{history_lesion_lists_str}\n"
        f"Current Structured Lesion List:\n{current_lesion_list_str}\n"
        "Please provide CALL or INELIGIBLE ( or FINAL)."
    )

    history_msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": base_user},
    ]

    def render_history() -> str:
        lines = ["【Dialogue with History】"]
        for i, m in enumerate(history_msgs, 1):
            lines.append(f"{i}. {m['role'].upper()}: {m['content']}")
        lines.append("Please output JSON only.")
        return "\n".join(lines)

    call_args: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        if DEBUG_PRINT:
            print(f"  [ASK] {attempt}/{max_attempts}th request (expecting CALL or INELIGIBLE)...")
        prompt = render_history()
        text = lightrag_query(prompt, "NAC Efficacy Assessment - Tool Evaluation")
        obj = extract_first_json(text)

        if not obj:
            history_msgs.append({"role": "assistant", "content": "(Invalid JSON)"})
            history_msgs.append({"role": "user", "content": "Please strictly output JSON only as per the agreement."})
            continue

        act = obj.get("action")
        if act == "CALL" and obj.get("tool") == "recist_calculator":
            call_args = obj.get("arguments") or {}
            if DEBUG_PRINT:
                print("    → 捕获 CALL：", json.dumps(call_args, ensure_ascii=False))
            history_msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
            break

        if act == "INELIGIBLE":
            if DEBUG_PRINT:
                print("    → Determined to be non-computable:")
            history_msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
            history_msgs.append({"role": "user", "content": "Please verify sentence by sentence whether there are truly no usable values (ignoring lymph nodes). If calculation is possible, perform CALL; if calculation remains impossible, continue returning INELIGIBLE."})
            continue

        if act == "FINAL":
            label = obj.get("recist_result")
            if label in ALLOWED_RECIST:
                if DEBUG_PRINT:
                    print("    → FINAL：", label)
                return (label, {"method": "gpt_direct_final", "first_final_obj": obj})
            history_msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
            history_msgs.append({"role": "user", "content": "If eligible for calculation, please CALL; if truly ineligible, please mark as INELIGIBLE and provide an explanation."})
            continue

        history_msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        history_msgs.append({"role": "user", "content": "Please return the CALL / INELIGIBLE / FINAL JSON from the agreement."})

    if call_args is None:
        return None

    tool_result = recist_calculator(**call_args)
    if DEBUG_PRINT:
        print("    → Tool execution results：", json.dumps(tool_result, ensure_ascii=False))

    tool_msg = {"tool": "recist_calculator", "result": tool_result}
    history_msgs.append({"role": "tool", "content": json.dumps(tool_msg, ensure_ascii=False)})
    if DEBUG_PRINT:
        print("    → Backfilled role=tool：", json.dumps(tool_msg, ensure_ascii=False))

    final_prompt = render_history() + "\n Please output JSON with action=FINAL based on the tool results."
    if DEBUG_PRINT:
        print("  [FINAL] Requesting FINAL JSON ...")
    final_text = lightrag_query(final_prompt, "NAC Efficacy Assessment - Final")
    final_obj = extract_first_json(final_text) or {}
    if DEBUG_PRINT:
        print("  [FINAL] Response：", json.dumps(final_obj, ensure_ascii=False))

    label = final_obj.get("recist_result")
    if label in ALLOWED_RECIST:
        return (label, {
            "method": "calculator",
            "call_args": call_args,
            "tool_result": tool_result,
            "final_obj": final_obj
        })
    return None


def _fallback_via_plain_rag(history_text: str,
                            current_desc: str,
                            history_lesion_lists_str: str,
                            current_lesion_list_str: str,
                            few_shot: str = "") -> str:
    examples_block = f"\n【Few-shot Examples (For Reference Only)】\n{few_shot}\n" if (few_shot and str(few_shot).strip()) else ""

    system_prompt = (
        "You are an expert in breast imaging and RECIST 1.1, conducting NAC efficacy assessment. Output only one label: Baseline/CR/PR/SD/PD."
        " When making the assessment, consider both the baseline and the changes since baseline, and ignore lymph node descriptions."
    )
    user_prompt = (
        f"{system_prompt}"
        f"{examples_block}\n"
        f"Historical description:\n{history_text}\n"
        f"Current description:\n{current_desc}\n"
        "Historical structured lesion list:\n"
        f"{history_lesion_lists_str}\n"
        "Current structured lesion list:\n"
        f"{current_lesion_list_str}\n\n"
        "Please output only one of the above five labels (Baseline/CR/PR/SD/PD), no explanations."
    )
    text = lightrag_query(user_prompt, "NAC Efficacy Assessment - Fallback")

def eval_recist_result(history_text: str,
                       current_desc: str,
                       history_lesion_lists_str: str,
                       current_lesion_list_str: str,
                       few_shot: str = "",
                       trace_id: Optional[str] = None) -> str:

    if _is_baseline_first_visit(history_text, history_lesion_lists_str):
        if DEBUG_PRINT:
            print("  [BASELINE] Detected first visit/baseline (no history), directly returning 'Baseline', not calling the calculator.")
        _log_decision(label="Baseline", method="baseline",
                      meta={"reason": "no_history"},
                      trace_id=trace_id)
        return "Baseline"

    ret = _recist_via_lightrag_tool(
        history_text=history_text,
        current_desc=current_desc,
        history_lesion_lists_str=history_lesion_lists_str,
        current_lesion_list_str=current_lesion_list_str,
        max_attempts=MAX_ATTEMPTS,
        few_shot=few_shot
    )
    if ret is not None:
        label, info = ret
        if label in ALLOWED_RECIST:
            _log_decision(label=label, method=info.get("method", "unknown"),
                          meta=info, trace_id=trace_id)
            return label

    if DEBUG_PRINT:
        print("  [FALLBACK] Path 1 did not yield a usable result, switching to fallback RAG interpretation.")
    label = _fallback_via_plain_rag(
        history_text=history_text,
        current_desc=current_desc,
        history_lesion_lists_str=history_lesion_lists_str,
        current_lesion_list_str=current_lesion_list_str,
        few_shot=few_shot
    )
    if label:
        _log_decision(label=label, method="gpt_fallback_plain",
                      meta=None, trace_id=trace_id)
    return label


