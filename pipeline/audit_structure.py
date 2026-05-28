import json, re

records = [json.loads(l) for l in open('data/train_sft_v3.jsonl', encoding='utf-8') if l.strip()]
print(f'Total records: {len(records)}')

# Inspect the structure of 5 records
print('\n=== RECORD STRUCTURE SAMPLES ===')
for i, r in enumerate(records[:5]):
    msgs = r.get('messages', [])
    meta = r.get('metadata', {})
    print(f'\nRecord {i}: metadata keys={list(meta.keys()) if meta else []}')
    for j, m in enumerate(msgs):
        role = m.get('role', '?')
        content = m.get('content', '') or ''
        # Summarise content
        if role == 'system':
            summary = f'[system prompt {len(content)} chars]'
        elif role == 'tool':
            summary = f'[tool result: {content[:80]}]'
        elif '<tool>' in content:
            tool_match = re.search(r'<tool>(.+?)</tool>', content, re.DOTALL)
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            think_len = len(think_match.group(1).strip()) if think_match else 0
            summary = f'[assistant: think={think_len}c, tool_call={tool_match.group(1)[:60] if tool_match else "?"}]'
        elif '<answer>' in content:
            ans_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            think_len = len(think_match.group(1).strip()) if think_match else 0
            ans_preview = ans_match.group(1).strip()[:60] if ans_match else '?'
            summary = f'[assistant FINAL: think={think_len}c, answer={ans_preview}]'
        else:
            summary = f'[{role}: {content[:80]}]'
        print(f'  msg[{j}] {role}: {summary}')

# Count multi-turn depths
print('\n=== CONVERSATION DEPTH DISTRIBUTION ===')
depths = {}
for r in records:
    n = len(r.get('messages', []))
    depths[n] = depths.get(n, 0) + 1
for k in sorted(depths.keys()):
    print(f'  {k} messages: {depths[k]} records')

# Check think block quality in FINAL assistant turn only
print('\n=== FINAL ASSISTANT TURN THINK BLOCK QUALITY ===')
final_empty = 0
final_short = 0
final_good = 0
final_no_think = 0
final_no_answer = 0
for r in records:
    msgs = r.get('messages', [])
    # Find the last assistant message
    last_asst = None
    for m in reversed(msgs):
        if m.get('role') == 'assistant':
            last_asst = m
            break
    if last_asst is None:
        continue
    content = last_asst.get('content', '') or ''
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if not think_match:
        final_no_think += 1
    else:
        t = think_match.group(1).strip()
        if len(t) == 0:
            final_empty += 1
        elif len(t) < 100:
            final_short += 1
        else:
            final_good += 1
    if '<answer>' not in content:
        final_no_answer += 1

print(f'  no think tag  : {final_no_think}')
print(f'  think empty   : {final_empty}')
print(f'  think short   : {final_short}')
print(f'  think good    : {final_good}')
print(f'  no answer tag : {final_no_answer}')

# Look at a record where final turn has no answer
print('\n=== SAMPLE RECORDS WHERE FINAL TURN HAS NO <answer> ===')
count = 0
for r in records:
    msgs = r.get('messages', [])
    last_asst = None
    for m in reversed(msgs):
        if m.get('role') == 'assistant':
            last_asst = m
            break
    if last_asst and '<answer>' not in (last_asst.get('content') or ''):
        print(f'\n  Final turn content: {(last_asst.get("content") or "")[:200]}')
        count += 1
        if count >= 3:
            break

# Check the role sequence pattern
print('\n=== ROLE SEQUENCE PATTERNS ===')
patterns = {}
for r in records:
    seq = tuple(m.get('role', '?') for m in r.get('messages', []))
    patterns[seq] = patterns.get(seq, 0) + 1
# Show top 10
for seq, cnt in sorted(patterns.items(), key=lambda x: -x[1])[:10]:
    print(f'  {cnt:4d}x  {" -> ".join(seq)}')

# Check if tool role messages exist (as separate messages, not in content)
print('\n=== TOOL RESULT MESSAGES (role=tool) ===')
tool_role_msgs = sum(1 for r in records for m in r.get('messages', []) if m.get('role') == 'tool')
print(f'  Messages with role=tool: {tool_role_msgs}')

# Show a full example with role=tool
for r in records[:50]:
    if any(m.get('role') == 'tool' for m in r.get('messages', [])):
        print('\n  Sample record with tool role messages:')
        for m in r['messages']:
            role = m.get('role')
            content = (m.get('content') or '')[:120]
            print(f'    [{role}] {content}')
        break
