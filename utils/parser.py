import re

def extract_max_diameter(text):

    if not text:
        return None

    match = re.search(r"(?:Major diameter | Maximum diameter | Maximum major diameter)[^\d]{0,5}(\d+(\.\d+)?)\s*(cm|mm)", text)
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if value == 0: return None
        return value * 10 if unit == "cm" else value

    match = re.search(r"((\d+(\.\d+)?)[\*×xX]+)+(\d+(\.\d+)?)(\s*)(cm|mm)", text.replace(" ", ""))
    if match:
        nums = re.findall(r"\d+(\.\d+)?", match.group(0))
        nums = [float(n) for n in nums if n and float(n) != 0]
        if not nums: return None
        unit = match.group(7)
        max_value = max(nums)
        return max_value * 10 if unit == "cm" else max_value

    match = re.search(r"(\d+(\.\d+)?)\s*(cm|mm)", text)
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if value == 0: return None
        return value * 10 if unit == "cm" else value

    return None

def extract_birads(text):

    if not text:
        return ""
    match = re.search(r"BI[\-\s]?RADS[^\d]{0,3}(\d([abcAB]?)?)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def is_baseline(record):

    visit_id = record.get("visit_id", "").lower()
    desc = record.get("description", "")
    if "baseline" in visit_id or "Initial diagnosis" in desc or "基线" in desc:
        return True
    return False
