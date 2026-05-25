import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8

with open(path, encoding='utf-8') as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        msgs = ex.get('messages', [])
        assistant_msgs = [m for m in msgs if m['role'] == 'assistant']
        last = assistant_msgs[-1]['content'] if assistant_msgs else ''

        if '<answer>' not in last:
            print(f"--- line {lineno} ---")
            meta = ex.get('metadata', {})
            print(f"type: {meta.get('type', meta.get('category','?'))}")
            q = next((m['content'][:100] for m in msgs if m['role']=='user'), '')
            print(f"Q: {q}")
            print(f"#turns: {len(assistant_msgs)}, last_len: {len(last)}")
            # show structure of last assistant turn
            think_match = re.search(r'<think>(.*?)</think>', last, re.DOTALL)
            after_think = re.sub(r'<think>.*?</think>', '', last, flags=re.DOTALL).strip()
            print(f"has_think: {bool(think_match)}, after_think_len: {len(after_think)}")
            print(f"after_think preview: {after_think[:200]}")
            print()
            limit -= 1
            if limit <= 0:
                break
