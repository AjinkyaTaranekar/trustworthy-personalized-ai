import json, sys, collections, re
sys.stdout.reconfigure(encoding='utf-8')

def inspect(path):
    total = 0
    first_tool_counter = collections.Counter()
    type_counter = collections.Counter()
    has_answer_in_last = 0
    has_answer_anywhere = 0
    no_final_answer = 0
    mem_update_examples = 0
    math_mem_examples = 0   # math question that starts with user_memory_sections
    tool_in_last_only = 0   # last assistant turn is just a tool call, no answer
    looping_content = 0     # tool content that repeats >50 chars of same substring

    examples_to_show = {'math_mem': [], 'no_answer': [], 'loop': []}

    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get('messages', [])
            meta = ex.get('metadata', {})
            qtype = meta.get('type', meta.get('category', 'unknown'))
            type_counter[qtype] += 1
            total += 1

            assistant_msgs = [m for m in msgs if m['role'] == 'assistant']
            if not assistant_msgs:
                continue

            first_asst = assistant_msgs[0]['content'].strip()
            last_asst  = assistant_msgs[-1]['content'].strip()

            # First tool
            tools = re.findall(r'<tool>(\w+)\(', first_asst)
            first_tool = tools[0] if tools else 'direct_answer'
            first_tool_counter[first_tool] += 1

            # Answer tag presence
            if '<answer>' in last_asst:
                has_answer_in_last += 1
            if any('<answer>' in m['content'] for m in assistant_msgs):
                has_answer_anywhere += 1
            else:
                no_final_answer += 1
                if len(examples_to_show['no_answer']) < 3:
                    q = next((m['content'][:80] for m in msgs if m['role']=='user'), '')
                    examples_to_show['no_answer'].append((lineno, q, last_asst[:200]))

            # user_memory_update in any assistant turn
            if any('user_memory_update' in m['content'] for m in assistant_msgs):
                mem_update_examples += 1

            # Math question starting with user_memory_sections
            is_math = qtype in {'math','arithmetic','algebra','geometry','statistics',
                                 'unit_conversion','word_problems','trigonometry',
                                 'calculus','advanced_geometry','computation'}
            if is_math and first_tool == 'user_memory_sections':
                math_mem_examples += 1
                if len(examples_to_show['math_mem']) < 3:
                    q = next((m['content'][:80] for m in msgs if m['role']=='user'), '')
                    examples_to_show['math_mem'].append((lineno, qtype, q, first_asst[:200]))

            # Looping content detection
            for m in assistant_msgs:
                c = m['content']
                if len(c) > 500:
                    # check for repeated 30-char chunks
                    chunk = c[50:80]
                    if chunk and c.count(chunk) > 5:
                        looping_content += 1
                        if len(examples_to_show['loop']) < 2:
                            examples_to_show['loop'].append((lineno, c[:150]))
                        break

    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print(f"{'='*60}")
    print(f"Total examples:          {total}")
    print(f"Has <answer> in last:    {has_answer_in_last} ({100*has_answer_in_last/total:.1f}%)")
    print(f"Has <answer> anywhere:   {has_answer_anywhere} ({100*has_answer_anywhere/total:.1f}%)")
    print(f"No <answer> at all:      {no_final_answer} ({100*no_final_answer/total:.1f}%)")
    print(f"Has user_memory_update:  {mem_update_examples} ({100*mem_update_examples/total:.1f}%)")
    print(f"Math+mem_sections start: {math_mem_examples}")
    print(f"Looping content:         {looping_content}")
    print(f"\nFirst tool distribution:")
    for t, c in first_tool_counter.most_common(10):
        print(f"  {t:<35} {c:>5} ({100*c/total:.1f}%)")
    print(f"\nQuestion types:")
    for t, c in type_counter.most_common(15):
        print(f"  {t:<35} {c:>5}")
    if examples_to_show['no_answer']:
        print(f"\nNo-answer examples:")
        for lineno, q, last in examples_to_show['no_answer']:
            print(f"  line {lineno}: Q={q}")
            print(f"  last_asst={last[:150]}")
    if examples_to_show['loop']:
        print(f"\nLooping examples:")
        for lineno, c in examples_to_show['loop']:
            print(f"  line {lineno}: {c}")

for path in sys.argv[1:]:
    inspect(path)
