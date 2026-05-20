import json

with open('pipeline/reports/constitution_probe_20260520_131522.json') as f:
    data = json.load(f)

print('SCORES:')
for k, v in data['scores_by_principle'].items():
    marker = 'PASS' if v == 1.0 else ('PART' if v > 0 else 'FAIL')
    print(f'  {marker} {k}: {v}')

print('\n=== FAILING PRINCIPLES ===')
for probe in data['probe_results']:
    pid = probe['id']
    score = probe['combined_score']
    if score < 1.0:
        print(f'\n--- {pid} (score={score}) ---')
        for qr in probe['question_results']:
            if qr['combined_score'] < 1.0:
                print(f'  Q: {qr["question"]}')
                resp = qr['response']
                # strip think block, show first 600 chars of actual response
                import re
                clean = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
                print(f'  R: {clean[:600]}')
                print()
