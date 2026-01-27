# -*- coding: utf-8 -*-
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from utils.chat import call_gpt_with_token



# ---------------------------------------------------------------------
# Pure numeric calculator (no label/category logic)
# ---------------------------------------------------------------------
def recist_calculator(
    baseline_sld_mm: Optional[float] = None,
    current_sld_mm: Optional[float] = None,
    nadir_sld_mm: Optional[float] = None,
    new_lesions: bool = False,
    non_target_pd: bool = False,
) -> Dict[str, Any]:
    """
    Pure numeric deltas only.
    """
    if current_sld_mm is None:
        return {"ok": False, "error": "missing current_sld_mm"}

    def _to_float(x):
        if x in (None, ""):
            return None
        try:
            return float(x)
        except Exception:
            return None

    cur = _to_float(current_sld_mm)
    base = _to_float(baseline_sld_mm)
    nadir = _to_float(nadir_sld_mm)

    if cur is None:
        return {"ok": False, "error": "invalid current_sld_mm"}

    delta_vs_baseline_pct = None
    if base not in (None, 0.0):
        delta_vs_baseline_pct = (cur - base) / base * 100.0

    delta_vs_nadir_pct = None
    if nadir not in (None, 0.0):
        delta_vs_nadir_pct = (cur - nadir) / nadir * 100.0

    return {
        "ok": True,
        "baseline_sld_mm": base,
        "nadir_sld_mm": nadir,
        "current_sld_mm": cur,
        "new_lesions": bool(new_lesions),
        "non_target_pd": bool(non_target_pd),
        "delta_vs_baseline_pct": delta_vs_baseline_pct,
        "delta_vs_nadir_pct": delta_vs_nadir_pct,
    }


# ---------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------
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


def _ensure_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(_strip_code_fences(x))
        except Exception:
            return {}
    return {}


def _looks_like_empty_structured_report(sr: Any) -> bool:
    if sr is None:
        return True
    if isinstance(sr, dict) and len(sr) == 0:
        return True
    if isinstance(sr, str):
        t = sr.strip()
        if t in ("", "[]", "{}", "null", "None"):
            return True
        try:
            obj = json.loads(_strip_code_fences(t))
            if isinstance(obj, (list, dict)) and len(obj) == 0:
                return True
        except Exception:
            pass
    return False


def _extract_lesion_list_from_structured_report(sr: Any) -> List[Dict[str, Any]]:
    """
    Tolerant extraction of lesion list from structured report.
    """
    d = _ensure_dict(sr)
    ll = (
        d.get("2 Radiology Findings", {}).get("2.4 Lesion List", None)
        or d.get("2 Radiology Findings", {}).get("2.4 Lesion list", None)
        or d.get("2.4 Lesion List", None)
        or d.get("2.4 Lesion list", None)
    )
    return ll if isinstance(ll, list) else []


# ---------------------------------------------------------------------
# Core: ONLY two structured reports in; ONLY tool_result out
# ---------------------------------------------------------------------
def eval_numeric_change_from_structured_reports(
    historical_structured_report: Any,
    current_structured_report: Any,
    few_shot: str = "",
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Input:
      - historical_structured_report: dict or JSON string
      - current_structured_report: dict or JSON string
    Output:
      - tool_result dict from recist_calculator (numeric deltas only)
      - if not computable: {"ok": False, "error": "...", "details": {...}}

    Notes:
      - No LightRAG.
      - No response-category/label words or outputs.
      - GPT is only used to produce CALL/INELIGIBLE JSON.
    """
    if _looks_like_empty_structured_report(historical_structured_report):
        return {"ok": False, "error": "missing historical_structured_report"}

    hist_sr = _ensure_dict(historical_structured_report)
    cur_sr = _ensure_dict(current_structured_report)

    history_lesion_list = _extract_lesion_list_from_structured_report(hist_sr)
    current_lesion_list = _extract_lesion_list_from_structured_report(cur_sr)

    history_lesion_lists_str = json.dumps(history_lesion_list, ensure_ascii=False, indent=2)
    current_lesion_list_str = json.dumps(current_lesion_list, ensure_ascii=False, indent=2)

    examples_block = f"\n[Examples (reference only)]\n{few_shot}\n" if (few_shot and str(few_shot).strip()) else ""

    base_prompt = (
        "*** Please reference the supplementary materials. ***\n\n"
        f"{examples_block}"
        "Historical structured lesion list (JSON):\n"
        f"{history_lesion_lists_str}\n\n"
        "Current structured lesion list (JSON):\n"
        f"{current_lesion_list_str}\n\n"
        "Task:\n"
        "If numeric values can be determined, output JSON only:\n"
        '{"action":"CALL","tool":"recist_calculator","arguments":{"baseline_sld_mm":...,"nadir_sld_mm":...,"current_sld_mm":...,"new_lesions":...,"non_target_pd":...}}\n'
        "If not possible, output JSON only:\n"
        '{"action":"INELIGIBLE","reason":"...","notes":"..."}\n'
    )

    last_obj: Optional[Dict[str, Any]] = None
    total_tokens = 0

    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += "\nReminder: output a single valid JSON object only."

        text, tok = call_gpt_with_token(prompt)
        total_tokens += tok

        obj = extract_first_json(text)
        last_obj = obj

        if not obj:
            continue

        if obj.get("action") == "CALL" and obj.get("tool") == "recist_calculator":
            call_args = obj.get("arguments") or {}
            tool_result = recist_calculator(**call_args)
            tool_result["_tokens"] = total_tokens  # optional
            return tool_result

        if obj.get("action") == "INELIGIBLE":
            return {
                "ok": False,
                "error": "ineligible",
                "details": obj,
                "_tokens": total_tokens,
            }

    return {
        "ok": False,
        "error": "no_valid_json_or_call",
        "details": last_obj or {},
        "_tokens": total_tokens,
    }



