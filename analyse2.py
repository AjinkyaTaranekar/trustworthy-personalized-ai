import json

with open('pipeline/reports/constitution_probe_20260520_131522.json') as f:
    data = json.load(f)

print("probe_id | score | has_tool | has_answer | response_start")
for probe in data['probe_results']:
    for qr in probe['question_results']:
        resp = qr['response']
        import re
        clean = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
        has_tool = '<tool>' in resp
        has_answer = '<answer>' in resp
        print(f"{probe['id']:30s} | {qr['combined_score']:.1f} | {str(has_tool):5} | {str(has_answer):5} | {clean[:80]}")
