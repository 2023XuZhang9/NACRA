import tiktoken

def get_encoding_for_model(model: str):
    model = model.lower()
    if "gpt-4" in model:
        return tiktoken.encoding_for_model("gpt-4")
    else:
        return tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    enc = get_encoding_for_model(model)
    return len(enc.encode(text))

def print_total_tokens(input_text, output_text, model="gpt-4"):
    total_tokens = estimate_tokens(input_text, model) + estimate_tokens(output_text, model)
    print(f"\n all token ：{total_tokens}")
