import json, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')

def find_tools(text):
    return re.findall(r'<tool>(\w+)\(', text)

def is_looping(text):
    for body in re.findall(r'<tool>.*?(?:</tool>|$)', text, re.DOTALL):
        if len(body) < 300:
            continue
        mid = len(body) // 2
        chunk = body[mid:mid+40].strip()
        if chunk and body.count(chunk) > 5:
            return True
    return False

for path in sys.argv[1:]:
    total = no_answer = has_mem_update = looping = 0
    first_tool = collections.Counter()

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get('messages', [])
            total += 1

            asst = [m for m in msgs if m['role'] == 'assistant']
            if not asst:
                continue

            first_tools = find_tools(asst[0]['content'])
            first_tool[first_tools[0] if first_tools else 'direct'] += 1

            last = asst[-1]['content']
            if '<answer>' not in last:
                no_answer += 1
            if any('user_memory_update' in m['content'] for m in asst):
                has_mem_update += 1
            if is_looping(' '.join(m['content'] for m in asst)):
                looping += 1

    print(f"\n{path}")
    print(f"  total={total}  no_answer={no_answer}  mem_update={has_mem_update}  looping={looping}")
    print(f"  first tool: {dict(first_tool.most_common(6))}")
