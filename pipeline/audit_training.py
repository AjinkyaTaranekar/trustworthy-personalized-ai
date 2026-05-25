import json, collections

path = 'data/train_sft_v3.jsonl'

tool_first = collections.Counter()
has_think = 0
has_answer_tag = 0
has_tool_in_response = 0
math_with_python = 0
math_without_python = 0
total = 0
multi_turn = 0
tool_call_patterns = collections.Counter()

math_cats = {'arithmetic','algebra','geometry','statistics','unit_conversion',
             'word_problems','trigonometry','calculus','advanced_geometry'}

with open(path, encoding='utf-8') as f:
    for line in f:
        ex = json.loads(line)
        msgs = ex.get('messages', [])
        meta = ex.get('metadata', {})
        cat = meta.get('category', '')
        total += 1

        # Count turns
        user_turns = sum(1 for m in msgs if m['role'] == 'user')
        if user_turns > 1:
            multi_turn += 1

        for m in msgs:
            if m['role'] == 'assistant':
                c = m['content'].strip()
                if '<think>' in c:
                    has_think += 1
                if '<answer>' in c:
                    has_answer_tag += 1
                if '<tool>' in c:
                    has_tool_in_response += 1
                    # what tool?
                    import re
                    tools = re.findall(r'<tool>(\w+)\(', c)
                    for t in tools:
                        tool_call_patterns[t] += 1
                    first_tool = tools[0] if tools else 'unknown'
                    tool_first[first_tool] += 1
                # math category checks
                if cat in math_cats:
                    if 'python_execute' in c:
                        math_with_python += 1
                    else:
                        math_without_python += 1
                break

print(f"Total examples: {total}")
print(f"Multi-turn: {multi_turn} ({100*multi_turn/total:.1f}%)")
print(f"Has <think>: {has_think} ({100*has_think/total:.1f}%)")
print(f"Has <answer>: {has_answer_tag} ({100*has_answer_tag/total:.1f}%)")
print(f"Has any <tool>: {has_tool_in_response} ({100*has_tool_in_response/total:.1f}%)")
print(f"\nTool call frequency (all turns):")
for t, c in tool_call_patterns.most_common(15):
    print(f"  {t}: {c}")
print(f"\nFirst tool called per example:")
for t, c in tool_first.most_common(10):
    print(f"  {t}: {c}")
print(f"\nMath category examples:")
print(f"  with python_execute: {math_with_python}")
print(f"  without python_execute: {math_without_python}")
