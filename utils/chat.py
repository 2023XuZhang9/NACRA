import requests

HOST = "https://openrouter.ai/api/v1"
KEY = "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"
MODEL = "gpt-4o"
def call_gpt_with_token(prompt: str) -> str:
    url = f"{HOST}/chat/completions"
    payload = {
        "model": MODEL,
        "max_tokens": 2048,
        "temperature": 0.001,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
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



