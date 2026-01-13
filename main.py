import os
import json
import requests
from report_processor import memory_checker
from utils.parser import is_baseline
from utils.tokentracker import estimate_tokens
from concurrent.futures import ThreadPoolExecutor, as_completed

need_STORE_DIR = "raw_data"
STORE_DIR = "records_updated"
os.makedirs(STORE_DIR, exist_ok=True)
OUTPUT_DIR = "structured_reports"
API_URL = "LIGHTRAG_URL"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TOKEN_MODEL = "gpt-4o"
role_desc = "You are a senior breast tumor imaging expert, familiar with breast MRI report interpretation, BI-RADS classification, and RECIST efficacy assessment. Please use the provided description as the sole source of information to conduct step-by-step reasoning and structured report writing, ensuring the output is objective, traceable, and verifiable, and based solely on the provided information, without conjecture."

def upsert_patient_records(patient_id: str, incoming_records):
    if not isinstance(incoming_records, list):
        incoming_records = [incoming_records] if incoming_records else []

    dst_path = os.path.join(STORE_DIR, f"{patient_id}.json")
    if os.path.exists(dst_path):
        with open(dst_path, "r", encoding="utf-8") as f:
            dst_json = json.load(f)
    else:
        dst_json = {"patient_id": patient_id, "records": []}

    existing_map = {}
    for rec in dst_json.get("records", []):
        vid = str(rec.get("visit_id", "")).strip()
        if vid:
            existing_map[vid] = rec

    added_cnt, updated_cnt = 0, 0

    for new_rec in incoming_records:
        if not isinstance(new_rec, dict):
            continue
        vid = str(new_rec.get("visit_id", "")).strip()
        if not vid:
            continue
        if vid in existing_map:
            changed = False
            tgt = existing_map[vid]
            for k, v in new_rec.items():
                if (k not in tgt) or (tgt[k] in [None, "", [], {}]):
                    tgt[k] = v
                    changed = True
            if changed:
                updated_cnt += 1
        else:
            dst_json["records"].append(new_rec)
            existing_map[vid] = new_rec
            added_cnt += 1

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(dst_json, f, ensure_ascii=False, indent=2)

    if added_cnt or updated_cnt:
        print(f"[Sync] {patient_id}: Added {added_cnt} entries, updated {updated_cnt} entries → {dst_path}")
    else:
        print(f"[Sync] {patient_id}: No updates needed")

    return dst_json

def pre_sync_from_source():
    src_files = sorted([f for f in os.listdir(need_STORE_DIR) if f.lower().endswith(".json")])
    if not src_files:
        print(f"[Sync] No JSON files found in source directory {need_STORE_DIR}.")
        return
    print(f"[Sync] Start merging patient data from {need_STORE_DIR} → {STORE_DIR}, {len(src_files)} files in total.")
    for fname in src_files:
        src_path = os.path.join(need_STORE_DIR, fname)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                src_json = json.load(f)
            patient_id = src_json.get("patient_id") or os.path.splitext(fname)[0]
            records = src_json.get("records", [])
            upsert_patient_records(patient_id, records)
        except Exception as e:
            print(f"[Sync] Merge failed: {fname} → {e}")

MEMORY_TO_TEMPLATE_MAP = {
    "Treatment Response Evaluation": ["4 Neoadjuvant Therapy Evaluation", "4.1 Treatment Response Evaluation"],
    "Radiology Diagnosis": ["3 Radiology Diagnosis"],
}


def clear_baseline_clinical_fields(struct_json: dict) -> dict:
    try:
        struct_json.setdefault("1 Clinical Information", {})
        struct_json["1 Clinical Information"].setdefault("1.1 Location", "")
        struct_json["1 Clinical Information"].setdefault("1.2 Other", "")
        struct_json["1 Clinical Information"]["1.1 Location"] = ""
        struct_json["1 Clinical Information"]["1.2 Other"] = ""
    except Exception:
        pass
    return struct_json

def load_patient_lesion_history(patient_id, folder="lesion_lists"):
    fpath = os.path.join(folder, f"{patient_id}.json")
    if not os.path.exists(fpath):
        return {
            "patient_id": patient_id,
            "lesion_history": []
        }
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_template_snippet():
    with open("template/structured_report_template.json", encoding="utf-8") as f:
        template = json.load(f)
    return template, json.dumps(template, ensure_ascii=False, indent=2)

def load_few_shot(shot_mode=0, agent_name="structured_report"):
    fname_map = {
        "structured_report": "shots/structured_report.json",
        "birads": "shots/birads.json",
        "diag": "shots/diag.json",
        "recist": "shots/recist.json"
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
        if isinstance(a, dict):
            a_str = json.dumps(a, ensure_ascii=False, indent=2)
        else:
            a_str = str(a)
        example_strs.append(f"Example:\nDescription: {q}\nOutput: {a_str}")
    return "\n\n".join(example_strs) + "\n\n" if example_strs else ""

def memory_to_partial_json(memory, mapping):
    partial_json = {}
    for k, v in memory.items():
        if v and v != "Missing" and k in mapping:
            ptr = partial_json
            path = mapping[k]
            for node in path[:-1]:
                if node not in ptr:
                    ptr[node] = {}
                ptr = ptr[node]
            ptr[path[-1]] = v
    return partial_json

def build_structured_prompt(template_snippet, desc, shot_examples, partial_json=None):
    partial_hint = ""
    if partial_json and len(partial_json) > 0:
        partial_hint = (
            "Below is the Partial JSON filled by AI. **Please retain the key fields exactly as they are in the output JSON result**. The remaining fields will be auto-filled strictly according to the template and description:\n"
            + json.dumps(partial_json, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    prompt = (
        "*** Please reference the supplementary materials. ***"
    )
    return prompt

def filter_json_by_template(output_json, template):
    if isinstance(template, dict):
        filtered = {}
        for k in template:
            if k in output_json:
                filtered[k] = filter_json_by_template(output_json[k], template[k])
            else:
                filtered[k] = template[k]
        return filtered
    elif isinstance(template, list):
        if isinstance(output_json, list):
            return output_json
        return template
    else:
        return output_json if output_json not in [None, "", [], {}] else template

def process_patient(patient_path, template_dict, template_snippet, sr_shot_examples, shot_mode):
    with open(patient_path, encoding="utf-8") as f:
        pdata = json.load(f)
    patient_id = pdata.get("patient_id") or os.path.splitext(os.path.basename(patient_path))[0]
    records = pdata.get("records", [])

    structured_records = []
    desc_list = []
    lesion_folder = "lesion_lists"
    lesion_history_json = load_patient_lesion_history(patient_id, lesion_folder)
    history_lesion_lists = lesion_history_json["lesion_history"]
    for idx, rec in enumerate(records):
        desc = rec.get("description", "")
        visit_id = rec.get("visit_id", "")
        combined_desc = (
            f"【Patient ID: {patient_id}】 Visit {idx + 1} (visit_id: {visit_id})\n"
            f"{desc}"
        )

        memory = {
            "patient_id": patient_id,
            "visit_id": visit_id,
            "history_descriptions": desc_list.copy(),
            "current_description": combined_desc,
            "history_lesion_lists": history_lesion_lists.copy(),
            "current_lesion_list": [],
            "Treatment Response Evaluation": "",
            "Radiology Diagnosis": "",
        }
        print(f"\n==== {patient_id} Visit {idx+1} ====")
        desc_list.append(combined_desc)
        is_base = (len(desc_list) == 1) or is_baseline(rec)
        memory, pre_tokens = memory_checker(memory, is_baseline_flag=is_base, shot_mode=shot_mode)
        if memory.get("current_lesion_list"):
            history_lesion_lists.append({
                "visit_id": visit_id,
                "Lesion List": memory["current_lesion_list"]
            })
        print(f"Accumulated Agent Token: {pre_tokens}")

        partial_json = memory_to_partial_json(memory, MEMORY_TO_TEMPLATE_MAP)
        current_desc = memory["current_description"]
        history_text = "\n\n====Historical Visit Descriptions====\n\n".join(memory["history_descriptions"])
        desc_for_struct = current_desc + (f"\n\n【Historical Descriptions for reference】\n{history_text}" if history_text else "")
        prompt = build_structured_prompt(template_snippet, desc_for_struct, sr_shot_examples, partial_json)
        data = {"query": desc_for_struct, "mode": "local", "user_prompt": prompt}
        resp = requests.post(API_URL, json=data)
        resp_txt = ""
        try:
            resp_txt = resp.json().get("response", "")
        except Exception:
            resp_txt = ""
        try:
            struct_json = json.loads(resp_txt)
        except Exception:
            import re
            match = re.search(r'\{[\s\S]*\}', resp_txt or "")
            struct_json = json.loads(match.group()) if match else {}

        for k, v in memory.items():
            if v and v != "Missing" and k in MEMORY_TO_TEMPLATE_MAP:
                ptr = struct_json
                path = MEMORY_TO_TEMPLATE_MAP[k]
                for node in path[:-1]:
                    if node not in ptr:
                        ptr[node] = {}
                    ptr = ptr[node]
                ptr[path[-1]] = v

        if memory.get("current_lesion_list"):
            struct_json.setdefault("2 Radiology Findings", {})
            struct_json["2 Radiology Findings"]["2.4 Lesion List"] = memory["current_lesion_list"]

        struct_json = filter_json_by_template(struct_json, template_dict)

        if is_base:
            struct_json = clear_baseline_clinical_fields(struct_json)
        input_text = prompt
        output_text = json.dumps(struct_json, ensure_ascii=False)
        token_count = estimate_tokens(input_text, model=TOKEN_MODEL) + estimate_tokens(output_text, model=TOKEN_MODEL)
        total_token_count = pre_tokens + token_count

        structured_records.append({
            "visit_id": visit_id,
            "description": desc,
            "structured_report": struct_json,
            "Token": total_token_count
        })
        print(f"{patient_id} {visit_id} Structured Report generated. Token:{total_token_count}")

    result = {
        "patient_id": patient_id,
        "records": structured_records
    }
    out_path = os.path.join(OUTPUT_DIR, f"{patient_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n====== Saved all structured results for {patient_id} to {out_path} ======\n")

def main(shot_mode=0, max_workers=8):
    template_dict, template_snippet = load_template_snippet()
    sr_shot_examples = load_few_shot(shot_mode, agent_name="structured_report")
    files = sorted([f for f in os.listdir(STORE_DIR) if f.endswith(".json")])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_patient,
                os.path.join(STORE_DIR, fname),
                template_dict,
                template_snippet,
                sr_shot_examples,
                shot_mode
            )
            for fname in files
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Processing failed: {e}")

if __name__ == "__main__":
    pre_sync_from_source()
    main(shot_mode=1, max_workers=1)
