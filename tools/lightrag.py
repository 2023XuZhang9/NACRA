import requests

API_URL = "http://localhost:9621/query"

def query_lightrag(prompt, context):
    payload = {
        "query": context,
        "mode": "mix",
        "user_prompt": prompt
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("answer") or data.get("content") or None
    except Exception as e:
        print(f"[LightRAG] fail: {e}")
        return None
