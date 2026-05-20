import json

count = 0
with open('pipeline/data/train_sft_v3.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        msgs = ex.get('messages', [])
        for m in msgs:
            content = m.get('content', '')
            if m['role'] == 'assistant' and 'python_execute' in str(content):
                print(json.dumps(m, indent=2)[:1200])
                print('---')
                count += 1
                if count >= 3:
                    break
        if count >= 3:
            break

print(f'found {count} examples with python_execute in assistant turn')
