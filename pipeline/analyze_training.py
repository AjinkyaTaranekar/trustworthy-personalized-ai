import json, sys, os

def analyze(path):
    counts = {'memory_update': 0, 'direct_answer': 0, 'other_tool': 0}
    tool_names = {}
    total = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex.get('messages', [])
            for msg in messages:
                if msg['role'] == 'assistant':
                    content = msg['content'].strip()
                    if '<tool>user_memory_update' in content[:80]:
                        counts['memory_update'] += 1
                    elif content.startswith('<tool>'):
                        counts['other_tool'] += 1
                        try:
                            name = content[6:content.find('(')]
                            tool_names[name] = tool_names.get(name, 0) + 1
                        except Exception:
                            pass
                    else:
                        counts['direct_answer'] += 1
                    total += 1
                    break

    print(f"\n{os.path.basename(path)}")
    print(f"  Total examples: {total}")
    for k, v in counts.items():
        pct = 100*v/total if total else 0
        print(f"  {k}: {v} ({pct:.1f}%)")
    if tool_names:
        print(f"  Other tools: {tool_names}")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        analyze(path)
