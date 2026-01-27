import requests

HOST = "xxx"
KEY = "xxx"
MODEL = "gpt-4o"
def call_gpt_with_token(prompt: str) -> str:
    url = f"{HOST}/chat/completions"
    payload = {
        "model": MODEL,
        "max_tokens": 2048,
        "temperature": 0.001,
        "messages": [
            {"role": "xxx", "content": prompt},
        ],
        "stream": False
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(data["choices"][0]["message"]["content"], data.get("usage", {}).get("total_tokens", 0))
    content = data["choices"][0]["message"]["content"]
    token = data.get("usage", {}).get("total_tokens", 0)
    return content, token




