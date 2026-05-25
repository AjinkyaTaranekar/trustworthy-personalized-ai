import json, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')

path = sys.argv[1] if len(sys.argv) > 1 else 'data/train_sft_v3.jsonl'

def find_tools(text):
    return re.findall(r'<tool>(\w+)\(', text)

total = 0
type_counter   = collections.Counter()
first_tool     = collections.Counter()   # first tool in first assistant turn
all_tools      = collections.Counter()   # every tool call across all turns
tool_sequences = collections.Counter()   # full sequence per example e.g. "user_memory_sections→user_memory_read→answer"
n_turns        = collections.Counter()   # number of assistant turns per example
has_answer     = 0
has_think      = 0
has_mem_update = 0
has_python     = 0
direct_answer  = 0   # no tool call at all
multi_turn_ex  = 0   # more than 1 assistant turn

# Per-type breakdown
type_first_tool = collections.defaultdict(collections.Counter)

with open(path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        msgs = ex.get('messages', [])
        meta = ex.get('metadata', {})
        qtype = meta.get('type', meta.get('category', 'unknown'))
        type_counter[qtype] += 1
        total += 1

        asst_msgs = [m for m in msgs if m['role'] == 'assistant']
        n_turns[len(asst_msgs)] += 1
        if len(asst_msgs) > 1:
            multi_turn_ex += 1

        # Collect all tool calls in order across all assistant turns
        seq = []
        for m in asst_msgs:
            seq.extend(find_tools(m['content']))
            if '<answer>' in m['content']:
                seq.append('__answer__')

        for t in seq:
            if t != '__answer__':
                all_tools[t] += 1

        if asst_msgs:
            first = asst_msgs[0]['content']
            ft = find_tools(first)
            fname = ft[0] if ft else 'direct_answer'
            first_tool[fname] += 1
            type_first_tool[qtype][fname] += 1

            last = asst_msgs[-1]['content']
            if '<answer>' in last:
                has_answer += 1
            if any('<think>' in m['content'] for m in asst_msgs):
                has_think += 1
            if any('user_memory_update' in m['content'] for m in asst_msgs):
                has_mem_update += 1
            if any('python_execute' in m['content'] for m in asst_msgs):
                has_python += 1
            if fname == 'direct_answer':
                direct_answer += 1

        # Sequence fingerprint (first 4 unique steps)
        seen = []
        for t in seq:
            if t not in seen:
                seen.append(t)
            if len(seen) == 4:
                break
        tool_sequences['→'.join(seen)] += 1

# ── Print report ──────────────────────────────────────────────────────────────

print(f"{'='*62}")
print(f"  DATASET ANALYSIS: {path}")
print(f"{'='*62}")
print(f"  Total examples   : {total}")
print(f"  Multi-turn       : {multi_turn_ex}  ({100*multi_turn_ex/total:.1f}%)")
print(f"  Has <think>      : {has_think}  ({100*has_think/total:.1f}%)")
print(f"  Has <answer>     : {has_answer}  ({100*has_answer/total:.1f}%)")
print(f"  Has mem_update   : {has_mem_update}  ({100*has_mem_update/total:.1f}%)")
print(f"  Has python_exec  : {has_python}  ({100*has_python/total:.1f}%)")
print(f"  Direct (no tool) : {direct_answer}  ({100*direct_answer/total:.1f}%)")

print(f"\n{'─'*62}")
print("  FIRST TOOL CALLED (per example)")
print(f"{'─'*62}")
for t, c in first_tool.most_common():
    bar = '█' * int(40 * c / total)
    print(f"  {t:<35} {c:>5}  {100*c/total:5.1f}%  {bar}")

print(f"\n{'─'*62}")
print("  ALL TOOL CALLS (total frequency across dataset)")
print(f"{'─'*62}")
grand = sum(all_tools.values())
for t, c in all_tools.most_common():
    bar = '█' * int(40 * c / grand)
    print(f"  {t:<35} {c:>6}  {100*c/grand:5.1f}%  {bar}")

print(f"\n{'─'*62}")
print("  TOP TOOL SEQUENCES (first 4 unique steps)")
print(f"{'─'*62}")
for seq, c in tool_sequences.most_common(12):
    print(f"  {c:>5}×  {seq}")

print(f"\n{'─'*62}")
print("  ASSISTANT TURNS PER EXAMPLE")
print(f"{'─'*62}")
for n, c in sorted(n_turns.items()):
    bar = '█' * int(30 * c / total)
    print(f"  {n} turn(s): {c:>5}  {bar}")

print(f"\n{'─'*62}")
print("  FIRST TOOL BY QUESTION TYPE (top 3 types per category)")
print(f"{'─'*62}")
for qtype, counts in sorted(type_first_tool.items(), key=lambda x: -sum(x[1].values())):
    n = sum(counts.values())
    breakdown = '  |  '.join(f"{t}: {c}" for t, c in counts.most_common(3))
    print(f"  {qtype:<35} (n={n:>4})  {breakdown}")
