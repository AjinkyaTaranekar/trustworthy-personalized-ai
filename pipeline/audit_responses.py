import json

# Look at what the model actually answered in the latest benchmark run
path = 'reports/constitution_probe_20260524_224530.json'

with open(path, encoding='utf-8') as f:
    data = json.load(f)

results = data.get('results', data.get('probe_results', []))
if not results:
    # try top-level list
    if isinstance(data, list):
        results = data

print(f"Total probes: {len(results)}\n")
print("="*70)
for r in results:
    pid = r.get('principle_id') or r.get('id', '?')
    score = r.get('combined_score', r.get('score', '?'))
    passed = r.get('passed', r.get('questions_passed', '?'))
    resp = r.get('response', '')
    # truncate
    resp_preview = resp[:300].replace('\n', ' ') if resp else '[no response field]'
    print(f"[{pid}] score={score} passed={passed}")
    print(f"  Response: {resp_preview}")
    print()
