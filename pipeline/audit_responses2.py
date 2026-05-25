import json

# Check actual structure of constitution probe JSON
path = 'reports/constitution_probe_20260524_224530.json'

with open(path, encoding='utf-8') as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
print()

# Find where responses actually live
for key, val in data.items():
    if isinstance(val, list) and len(val) > 0:
        print(f"Key '{key}' is a list of {len(val)}, first item keys: {list(val[0].keys()) if isinstance(val[0], dict) else type(val[0])}")
    elif isinstance(val, dict):
        print(f"Key '{key}' is a dict with keys: {list(val.keys())[:8]}")
    else:
        print(f"Key '{key}': {repr(val)[:80]}")
