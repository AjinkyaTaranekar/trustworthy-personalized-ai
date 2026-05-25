import json

with open('data/train_sft_v3.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        msgs = ex.get('messages', [])
        for m in msgs:
            if m['role'] == 'assistant':
                print(f"Example {i}:")
                print(repr(m['content'][:400]))
                print()
                break
        if i >= 6:
            break
