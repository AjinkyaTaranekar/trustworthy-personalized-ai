import json, sys

sys.stdout.reconfigure(encoding='utf-8')

path = 'reports/constitution_probe_20260524_224530.json'
with open(path, encoding='utf-8') as f:
    data = json.load(f)

for probe in data['probe_results']:
    pid = probe['id']
    score = probe['combined_score']
    for qr in probe.get('question_results', []):
        q = qr.get('question', '')
        resp = qr.get('response', '')
        q_str = q if isinstance(q, str) else str(q)[:100]
        resp_preview = resp[:350].replace('\n', ' ') if resp else '[empty]'
        print(f"[{pid}] score={score}")
        print(f"  Q: {q_str[:100]}")
        print(f"  A: {resp_preview}")
        print()
