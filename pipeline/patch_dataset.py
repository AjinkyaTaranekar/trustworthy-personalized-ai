"""
Patch train_partA_v3.jsonl and train_partB_v3.jsonl in-place (writes to *_patched.jsonl).

Issues fixed:
  1. Missing <answer> tags  — wrap post-</think> content in <answer>...</answer>
  2. Looping tool content   — drop examples where any tool arg repeats >5× (runaway generation)
  3. Incomplete examples    — drop examples where last assistant turn has no real content
  4. user_memory_update in math questions — strip those tool-call turns from the conversation
"""

import json, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')

MATH_TYPES = {
    'math', 'arithmetic', 'algebra', 'geometry', 'statistics',
    'unit_conversion', 'word_problems', 'trigonometry', 'calculus',
    'advanced_geometry', 'computation', 'math_word_problems',
    'math_algebra', 'math_arithmetic', 'math_statistics',
    'math_trigonometry', 'math_geometry',
}

def find_tools(text):
    """Return list of tool names called in text using <tool>name( pattern."""
    return re.findall(r'<tool>(\w+)\(', text)

def is_looping(text):
    """Detect runaway repeated content inside a tool argument."""
    # Find all tool call bodies
    for body in re.findall(r'<tool>.*?(?:</tool>|$)', text, re.DOTALL):
        if len(body) < 300:
            continue
        # Take a 40-char window from the middle and count repeats
        mid = len(body) // 2
        chunk = body[mid:mid + 40].strip()
        if chunk and body.count(chunk) > 5:
            return True
    return False

def wrap_answer(last_content):
    """
    If content has <think>...</think> followed by real text, wrap that text
    in <answer>...</answer>.  If nothing follows </think>, the example is
    incomplete — return None to signal it should be dropped.
    """
    # Already has answer tag — nothing to do
    if '<answer>' in last_content:
        return last_content

    after = re.sub(r'<think>.*?</think>', '', last_content, flags=re.DOTALL).strip()
    # Also strip leftover tool-result blocks
    after = re.sub(r'\[TOOL_RESULT\].*?\[/TOOL_RESULT\]', '', after, flags=re.DOTALL).strip()
    # Strip bare tool calls with no answer text
    after_no_tools = re.sub(r'<tool>.*?</tool>', '', after, flags=re.DOTALL).strip()

    if len(after_no_tools) < 20:
        # No real answer content — drop this example
        return None

    return re.sub(r'(<think>.*?</think>)', r'\1', last_content, flags=re.DOTALL).rstrip() \
           + f'\n<answer>{after_no_tools}</answer>'

def strip_memory_update_turns(messages):
    """
    Remove assistant messages that consist solely of user_memory_update calls
    (with no real answer content) from math examples.
    """
    cleaned = []
    for msg in messages:
        if msg['role'] != 'assistant':
            cleaned.append(msg)
            continue
        content = msg['content']
        tools = find_tools(content)
        # Keep if it has an answer or is not purely a memory update
        if 'user_memory_update' not in tools or '<answer>' in content:
            cleaned.append(msg)
            continue
        # Pure memory-update turn with no answer — skip it
        after = re.sub(r'<tool>.*?</tool>', '', content, flags=re.DOTALL).strip()
        after = re.sub(r'<think>.*?</think>', '', after, flags=re.DOTALL).strip()
        if len(after) > 20:
            cleaned.append(msg)  # has real text too, keep it
        # else: pure tool call, drop the turn
    return cleaned

def patch_file(src_path, dst_path):
    stats = collections.Counter()
    written = 0

    with open(src_path, encoding='utf-8') as fin, \
         open(dst_path, 'w', encoding='utf-8') as fout:

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get('messages', [])
            meta = ex.get('metadata', {})
            qtype = meta.get('type', meta.get('category', ''))
            is_math = qtype in MATH_TYPES

            # ── 1. Drop looping examples ──────────────────────────────────
            full_text = ' '.join(m['content'] for m in msgs)
            if is_looping(full_text):
                stats['dropped_looping'] += 1
                continue

            # ── 2. Strip inappropriate user_memory_update turns (math) ────
            if is_math:
                before = len(msgs)
                msgs = strip_memory_update_turns(msgs)
                if len(msgs) < before:
                    stats['stripped_mem_update'] += 1

            # ── 3. Fix missing <answer> on last assistant turn ────────────
            asst_msgs = [m for m in msgs if m['role'] == 'assistant']
            if asst_msgs and '<answer>' not in asst_msgs[-1]['content']:
                fixed = wrap_answer(asst_msgs[-1]['content'])
                if fixed is None:
                    stats['dropped_no_answer'] += 1
                    continue
                # Patch the last assistant message in-place
                for i in range(len(msgs) - 1, -1, -1):
                    if msgs[i]['role'] == 'assistant':
                        msgs[i] = dict(msgs[i], content=fixed)
                        break
                stats['fixed_answer_tag'] += 1

            # ── 4. Drop if still no assistant messages ────────────────────
            if not any(m['role'] == 'assistant' for m in msgs):
                stats['dropped_empty'] += 1
                continue

            ex['messages'] = msgs
            fout.write(json.dumps(ex, ensure_ascii=False) + '\n')
            written += 1

    print(f"\n{src_path}  →  {dst_path}")
    print(f"  Written:              {written}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<30} {v}")

if __name__ == '__main__':
    patch_file('data/train_partA_v3.jsonl', 'data/train_partA_v3_patched.jsonl')
    patch_file('data/train_partB_v3.jsonl', 'data/train_partB_v3_patched.jsonl')
